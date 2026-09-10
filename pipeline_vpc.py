"""VAP Sales Forecasting Pipeline - VPC Configuration
 
ML pipeline for training and evaluating 4 forecasting models:

- Barrage Champion/Challenger

- Grounded Champion/Challenger

"""

import os

import sagemaker

from sagemaker.sklearn.estimator import SKLearn

from sagemaker.sklearn.processing import SKLearnProcessor

from sagemaker.network import NetworkConfig

from sagemaker.processing import ProcessingInput, ProcessingOutput

from sagemaker.workflow.pipeline import Pipeline

from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker.workflow.steps import TrainingStep, ProcessingStep

from sagemaker.workflow.properties import PropertyFile

from sagemaker.workflow.condition_step import ConditionStep

from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo

from sagemaker.workflow.functions import JsonGet, Join

from sagemaker.workflow.model_step import ModelStep

from sagemaker.inputs import TrainingInput

from sagemaker.model import Model

from config import get_config

# Load configuration
config = get_config()
REGION = config['region']

BUCKET = config['s3_bucket']

ROLE_ARN = config['role_arn']

RAW_DATA_S3_URI = f"s3://{BUCKET}/raw-data/"

PREPROCESS_OUTPUT_S3_PREFIX = f"s3://{BUCKET}/pipeline-output/preprocess"

TRAINING_OUTPUT_S3_PREFIX = f"s3://{BUCKET}/pipeline-output/train"

MIN_R2_THRESHOLD = float(os.environ.get("VAP_MIN_R2_THRESHOLD", "0.7"))
 
# VPC Configuration
VPC_SUBNETS = config['vpc_subnets']

VPC_SECURITY_GROUP_IDS = config['vpc_security_groups']

PIPELINE_NETWORK_CONFIG = NetworkConfig(

    subnets=VPC_SUBNETS,

    security_group_ids=VPC_SECURITY_GROUP_IDS,

    enable_network_isolation=False

)


PIPELINE_ENV = {
    "VAP_S3_BUCKET": BUCKET,
    "VAP_REGION": REGION,
}

 
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

pipeline_session = PipelineSession()


def _upload_code_deps(file_list, s3_key):
    """Packages the given extra Python files into a tarball and uploads to a fixed S3
    location, returning its URI -- used to bundle extra modules (train_with_holdout.py,
    inference.py, dashboard/*.py) alongside a plain SKLearnProcessor step. Needed
    because SKLearnProcessor.run() does not accept source_dir in this SDK version
    (confirmed directly: "TypeError: ScriptProcessor.run() got an unexpected keyword
    argument 'source_dir'") -- staying within SKLearnProcessor only, per direct
    instruction, rather than switching to FrameworkProcessor.

    IMPORTANT: ProcessingInput copies this tarball as-is, it does NOT auto-extract it
    -- the receiving script must extract it itself and add its directory to sys.path
    before importing anything from it (see repack_evaluation.py /
    merge_and_generate_dashboard.py for the extraction side of this)."""
    import boto3
    import tarfile
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        with tarfile.open(tmp.name, "w:gz") as tar:
            for f in file_list:
                tar.add(f, arcname=os.path.basename(f))
        boto3.client("s3").upload_file(tmp.name, BUCKET, s3_key)
    return f"s3://{BUCKET}/{s3_key}"

 
def build_preprocessing_step():

    processor = SKLearnProcessor(

        framework_version="1.0-1",

        role=ROLE_ARN,

        instance_type=os.environ.get("VAP_PREPROCESS_INSTANCE_TYPE", "ml.m5.large"),

        instance_count=1,

        sagemaker_session=pipeline_session,

        base_job_name="preprocess",

        env=PIPELINE_ENV,

        network_config=PIPELINE_NETWORK_CONFIG

    )

    return ProcessingStep(

        name="PreprocessData",

        processor=processor,

        inputs=[ProcessingInput(source=RAW_DATA_S3_URI, destination="/opt/ml/processing/input")],

        outputs=[ProcessingOutput(output_name="train", source="/opt/ml/processing/output", destination=PREPROCESS_OUTPUT_S3_PREFIX)],

        code="preprocess.py",

        job_arguments=["--install-deps", "openpyxl"]

    )
 
def build_training_steps(preprocess_step):

    """Build training steps for all model variants"""

    models = [

        ("Barrage", "champion"),

        ("Barrage", "challenger"),

        ("Grounded", "champion"),

        ("Grounded", "challenger")

    ]

    training_steps = []

    for family, role in models:

        estimator = SKLearn(

            entry_point="train_with_holdout.py",

            source_dir=SOURCE_DIR,

            framework_version="1.0-1",

            role=ROLE_ARN,

            instance_type=os.environ.get("VAP_TRAIN_INSTANCE_TYPE", "ml.m5.xlarge"),

            instance_count=1,

            output_path=f"{TRAINING_OUTPUT_S3_PREFIX}/{family.lower()}-{role}",

            sagemaker_session=pipeline_session,

            hyperparameters={"family": family, "role": role},

            dependencies=["requirements.txt"],

            subnets=VPC_SUBNETS,

            security_group_ids=VPC_SECURITY_GROUP_IDS,

            environment=PIPELINE_ENV

        )

        step = TrainingStep(

            name=f"Train{family}{role.capitalize()}",

            estimator=estimator,

            inputs={"training": TrainingInput(s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri)}

        )

        training_steps.append(step)

    return training_steps
 
def build_repack_steps(training_steps, preprocess_step):

    """Extract Scenario 1's evaluation.json (UNCHANGED) -- AND train Scenario 2 (full
    data, no holdout), package it for registration, and generate this family/role's
    2-year forecast chunk using that SAME model object (see repack_evaluation.py)."""

    models = [

        ("Barrage", "champion"),

        ("Barrage", "challenger"),

        ("Grounded", "champion"),

        ("Grounded", "challenger")

    ]

    repack_steps = []

    for i, (family, role) in enumerate(models):

        processor = SKLearnProcessor(

            framework_version="1.0-1",

            role=ROLE_ARN,

            # UPGRADED from ml.m5.large: this step now also trains a full model on the
            # complete dataset and generates a 2-year recursive forecast, real compute
            # work beyond the original "extract one JSON file" job.
            instance_type=os.environ.get("VAP_REPACK_INSTANCE_TYPE", "ml.c5.4xlarge"),

            instance_count=1,

            sagemaker_session=pipeline_session,

            env=PIPELINE_ENV,

            network_config=PIPELINE_NETWORK_CONFIG

        )

        code_deps_uri = _upload_code_deps(

            [os.path.join(SOURCE_DIR, "train_with_holdout.py"),

             os.path.join(SOURCE_DIR, "inference.py"),

             os.path.join(SOURCE_DIR, "requirements.txt")],

            "pipeline-artifacts/repack_code_deps.tar.gz"

        )

        step_args = processor.run(

            code="repack_evaluation.py",

            inputs=[ProcessingInput(

                source=training_steps[i].properties.ModelArtifacts.S3ModelArtifacts,

                destination="/opt/ml/processing/input/scenario1_model"

            ), ProcessingInput(

                # NEW: the same preprocessed data Scenario 1 trained from, needed
                # here so Scenario 2 can train on the SAME source, just the full
                # dataset instead of the holdout split.
                source=preprocess_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,

                destination="/opt/ml/processing/input/data"

            ), ProcessingInput(

                # NEW: train_with_holdout.py + inference.py + requirements.txt, bundled
                # manually (see _upload_code_deps) since source_dir isn't supported here.
                # repack_evaluation.py extracts this tarball itself at startup.
                source=code_deps_uri,

                destination="/opt/ml/processing/input/code_deps"

            )],

            outputs=[ProcessingOutput(

                output_name="evaluation",

                source="/opt/ml/processing/output/evaluation",

                destination=f"{TRAINING_OUTPUT_S3_PREFIX}/evaluations/{family.lower()}-{role}"

            ), ProcessingOutput(

                # NEW: Scenario 2's packaged model.tar.gz, for registration.
                output_name="model",

                source="/opt/ml/processing/output/model",

                destination=f"{TRAINING_OUTPUT_S3_PREFIX}/scenario2-models/{family.lower()}-{role}"

            ), ProcessingOutput(

                # NEW: this family/role's forecast chunk, combined by the merge step.
                output_name="forecast",

                source="/opt/ml/processing/output/forecast",

                destination=f"s3://{BUCKET}/comparison/scenario2-chunks/{family.lower()}-{role}"

            ), ProcessingOutput(

                # NEW: this family/role's global feature importance chunk (top 30
                # highest-volume counties' sample), combined by the merge step into
                # feature_importance_global.csv -- an explicit customer requirement.
                output_name="importance",

                source="/opt/ml/processing/output/importance",

                destination=f"s3://{BUCKET}/comparison/scenario2-importance/{family.lower()}-{role}"

            )],

            arguments=["--family", family, "--role", role],

        )

        step = ProcessingStep(

            name=f"Repack{family}{role.capitalize()}",

            step_args=step_args,

        )

        repack_steps.append(step)

    return repack_steps
 
def build_evaluation_step(repack_steps):

    """Aggregate evaluation metrics from Scenario 1's per-role JSONs and check
    against the R2 quality threshold -- back to ONLY this, per direct instruction.
    Forecast generation, dashboard merge, and county accuracy all moved to
    Repack{Family}{Role} and the new MergeAndGenerateDashboard step."""

    processor = SKLearnProcessor(

        framework_version="1.0-1",

        role=ROLE_ARN,

        # DOWNGRADED back to ml.m5.large: this step is genuinely lightweight again --
        # just reading 4 small JSON files and aggregating them, no training or
        # forecast generation happens here anymore.
        instance_type=os.environ.get("VAP_EVALUATE_INSTANCE_TYPE", "ml.m5.large"),

        instance_count=1,

        sagemaker_session=pipeline_session,

        env=PIPELINE_ENV,

        network_config=PIPELINE_NETWORK_CONFIG

    )

    inputs = []

    models = [("Barrage", "champion", 0), ("Barrage", "challenger", 1), ("Grounded", "champion", 2), ("Grounded", "challenger", 3)]

    for i, (family, role, _) in enumerate(models):

        inputs.append(ProcessingInput(

            source=repack_steps[i].properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,

            destination=f"/opt/ml/processing/input/eval/{family.lower()}-{role}"

        ))

    # BUG FIX: this previously had "code=SOURCE_DIR" passed directly to ProcessingStep
    # -- SOURCE_DIR is a directory path, not a script filename, and ProcessingStep does
    # not accept source_dir directly at all (only through processor.run() as
    # step_args, same fix applied elsewhere in this file). This would have failed at
    # actual runtime. evaluate.py no longer imports inference.py or calls
    # dashboard_merge.py, so it doesn't need source_dir bundling at all anymore --
    # a plain code= is correct and sufficient here now.
    step_args = processor.run(

        code="evaluate.py",

        inputs=inputs,

        outputs=[ProcessingOutput(

            output_name="evaluation",

            source="/opt/ml/processing/evaluation",

            destination=f"{TRAINING_OUTPUT_S3_PREFIX}/final-evaluation"

        )],

    )

    step = ProcessingStep(

        name="EvaluateAllModels",

        step_args=step_args,

        property_files=[PropertyFile(

            name="EvaluationReport",

            output_name="evaluation",

            path="evaluation.json"

        )]

    )

    return step
 
def build_merge_step(repack_steps):

    """NEW, dedicated step (for clarity, per direct instruction) that combines the 4
    forecast chunks Repack generated into one forecast_2yr_all_models.csv, then runs
    dashboard_merge.py and generate_county_accuracy.py -- see
    merge_and_generate_dashboard.py for the full logic and defensive file checks."""

    processor = SKLearnProcessor(

        framework_version="1.0-1",

        role=ROLE_ARN,

        instance_type=os.environ.get("VAP_MERGE_INSTANCE_TYPE", "ml.m5.xlarge"),

        instance_count=1,

        sagemaker_session=pipeline_session,

        env=PIPELINE_ENV,

        network_config=PIPELINE_NETWORK_CONFIG

    )

    models = [("Barrage", "champion"), ("Barrage", "challenger"), ("Grounded", "champion"), ("Grounded", "challenger")]

    inputs = []

    for i, (family, role) in enumerate(models):

        inputs.append(ProcessingInput(

            source=repack_steps[i].properties.ProcessingOutputConfig.Outputs["forecast"].S3Output.S3Uri,

            destination=f"/opt/ml/processing/input/forecast_chunks/{family.lower()}-{role}"

        ))

        inputs.append(ProcessingInput(

            # NEW: this family/role's global feature importance chunk.
            source=repack_steps[i].properties.ProcessingOutputConfig.Outputs["importance"].S3Output.S3Uri,

            destination=f"/opt/ml/processing/input/importance_chunks/{family.lower()}-{role}"

        ))

    code_deps_uri = _upload_code_deps(

        [os.path.join(SOURCE_DIR, "dashboard", "dashboard_merge.py"),

         os.path.join(SOURCE_DIR, "dashboard", "generate_county_accuracy.py"),

         os.path.join(SOURCE_DIR, "requirements.txt")],

        "pipeline-artifacts/merge_code_deps.tar.gz"

    )

    inputs.append(ProcessingInput(

        # NEW: dashboard_merge.py + generate_county_accuracy.py + requirements.txt,
        # bundled manually (see _upload_code_deps) since source_dir isn't supported
        # here. merge_and_generate_dashboard.py extracts this tarball itself.
        source=code_deps_uri,

        destination="/opt/ml/processing/input/code_deps"

    ))

    step_args = processor.run(

        code="merge_and_generate_dashboard.py",

        inputs=inputs,

        outputs=[ProcessingOutput(

            output_name="merge_summary",

            source="/opt/ml/processing/output",

            destination=f"{TRAINING_OUTPUT_S3_PREFIX}/merge-summary"

        )],

    )

    step = ProcessingStep(

        name="MergeAndGenerateDashboard",

        step_args=step_args,
        # Explicit, even though the ProcessingInputs above already create this
        # dependency via their .properties references -- makes the real requirement
        # (all 4 Repack steps must finish first) unambiguous to read directly.
        depends_on=repack_steps,

    )

    return step
 
def build_quality_check_step(evaluation_step):

    """Validate model quality against threshold"""

    processor = SKLearnProcessor(

        framework_version="1.0-1",

        role=ROLE_ARN,

        instance_type=os.environ.get("VAP_QUALITY_CHECK_INSTANCE_TYPE", "ml.m5.large"),

        instance_count=1,

        sagemaker_session=pipeline_session,

        env=PIPELINE_ENV,

        network_config=PIPELINE_NETWORK_CONFIG

    )

    quality_report = PropertyFile(

        name="QualityReport",

        output_name="quality",

        path="quality_check.json"

    )

    step = ProcessingStep(

        name="QualityCheck",

        processor=processor,

        inputs=[ProcessingInput(

            source=evaluation_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,

            destination="/opt/ml/processing/input"

        )],

        outputs=[ProcessingOutput(

            output_name="quality",

            source="/opt/ml/processing/output",

            destination=f"{TRAINING_OUTPUT_S3_PREFIX}/quality-check"

        )],

        code="quality_check.py",

        job_arguments=["--r2-threshold", str(MIN_R2_THRESHOLD)],

        property_files=[quality_report]

    )

    return step, quality_report
 
def build_conditional_registration_steps(training_steps, repack_steps, evaluation_step):

    """Register the Scenario 2 model (from Repack, trained on the full dataset) that
    meets the quality threshold -- NOT Scenario 1 (from Training), per direct
    instruction: the model that gets deployed and the model that generated the
    forecast must be the same one. image_uri still comes from the training step
    (just the serving container/framework image, not tied to which scenario's
    weights are inside); model_data now comes from the repack step's Scenario 2
    output instead."""

    models = [

        ("Barrage", "champion", 0),

        ("Barrage", "challenger", 1),

        ("Grounded", "champion", 2),

        ("Grounded", "challenger", 3)

    ]

    condition_steps = []

    for family, role, idx in models:

        model = Model(

            image_uri=training_steps[idx].properties.AlgorithmSpecification.TrainingImage,

            # CHANGED: Scenario 2's model.tar.gz (from Repack), not Scenario 1's (from
            # Training) -- Join concatenates the ProcessingOutput's S3 directory with
            # the actual filename repack_evaluation.py writes inside it.
            model_data=Join(on="/", values=[

                repack_steps[idx].properties.ProcessingOutputConfig.Outputs["model"].S3Output.S3Uri,

                "model.tar.gz"

            ]),

            role=ROLE_ARN,

            sagemaker_session=pipeline_session

        )

        register_step = ModelStep(

            name=f"Register{family}{role.capitalize()}",

            step_args=model.register(

                content_types=["application/json"],

                response_types=["application/json"],

                inference_instances=["ml.m5.large"],

                transform_instances=["ml.m5.large"],

                model_package_group_name=f"vap-{family.lower()}-{role}",

                approval_status="PendingManualApproval"

            )

        )

        condition = ConditionStep(

            name=f"Check{family}{role.capitalize()}Quality",

            conditions=[ConditionGreaterThanOrEqualTo(

                left=JsonGet(

                    step_name=evaluation_step.name,

                    property_file=PropertyFile(

                        name="EvaluationReport",

                        output_name="evaluation",

                        path="evaluation.json"

                    ),

                    json_path=f"{family.lower()}_{role}.r2"

                ),

                right=MIN_R2_THRESHOLD

            )],

            if_steps=[register_step],

            else_steps=[]

        )

        condition_steps.append(condition)

    return condition_steps
 
if __name__ == "__main__":

    preprocess_step = build_preprocessing_step()

    training_steps = build_training_steps(preprocess_step)

    repack_steps = build_repack_steps(training_steps, preprocess_step)

    evaluation_step = build_evaluation_step(repack_steps)

    merge_step = build_merge_step(repack_steps)

    condition_steps = build_conditional_registration_steps(training_steps, repack_steps, evaluation_step)

    all_steps = ([preprocess_step] + training_steps + repack_steps
                 + [evaluation_step, merge_step] + condition_steps)

    pipeline = Pipeline(

        name="vap-forecast-pipeline-v4",

        steps=all_steps,

        sagemaker_session=pipeline_session

    )

    pipeline.upsert(role_arn=ROLE_ARN)

    print("Pipeline 'vap-forecast-pipeline-v4' created")

    print("Steps:")

    print("  1. PreprocessData")

    print("  2-5. Train (4 individual boxes) -- Scenario 1, holdout-trained, used for validation/quality only:")

    print("      - TrainBarrageChampion")

    print("      - TrainBarrageChallenger")

    print("      - TrainGroundedChampion")

    print("      - TrainGroundedChallenger")

    print("  6-9. Repack (4 individual boxes) -- extracts Scenario 1 eval, trains Scenario 2")
    print("       (full data, no holdout), generates that family/role's forecast chunk:")

    print("      - RepackBarrageChampion")

    print("      - RepackBarrageChallenger")

    print("      - RepackGroundedChampion")

    print("      - RepackGroundedChallenger")

    print("  10. EvaluateAllModels (Scenario 1 R2/quality check only)")

    print("  11. MergeAndGenerateDashboard (combines the 4 forecast chunks, runs")
    print("      dashboard_merge.py and generate_county_accuracy.py)")

    print("  12-15. Conditional Registration (4 individual) -- registers Scenario 2:")

    print("      - CheckBarrageChampionQuality → RegisterBarrageChampion (if R² >= 0.70)")

    print("      - CheckBarrageChallengerQuality → RegisterBarrageChallenger (if R² >= 0.70)")

    print("      - CheckGroundedChampionQuality → RegisterGroundedChampion (if R² >= 0.70)")

    print("      - CheckGroundedChallengerQuality → RegisterGroundedChallenger (if R² >= 0.70)")

    print(f"\nMin R² threshold: {MIN_R2_THRESHOLD}")

    print(f"Model registries: vap-barrage-champion, vap-barrage-challenger, vap-grounded-champion, vap-grounded-challenger")

    execution = pipeline.start()

    print(f"\nStarted execution: {execution.arn}")
 
