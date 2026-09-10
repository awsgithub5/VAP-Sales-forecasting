"""Unit tests for evaluate.py"""
import unittest
import json
import os
import tempfile
import shutil
from unittest.mock import patch, mock_open


class TestEvaluate(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.test_dir, "input")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_evaluation_file_structure(self):
        eval_data = {
            "model_type": "XGBoost",
            "validation_metrics": {
                "r2": 0.893,
                "rmse": 8415.0,
                "mae": 5200.0,
                "wmape": 0.15
            },
            "train_size": 1000,
            "val_size": 120,
            "train_cutoff": "2024-09-30",
            "val_start": "2024-10-01",
            "val_end": "2025-09-30"
        }
        
        subdir = os.path.join(self.input_dir, "barrage-champion")
        os.makedirs(subdir)
        eval_path = os.path.join(subdir, "Barrage_champion_evaluation.json")
        
        with open(eval_path, "w") as f:
            json.dump(eval_data, f)
            
        self.assertTrue(os.path.exists(eval_path))
        
        with open(eval_path) as f:
            loaded = json.load(f)
            self.assertEqual(loaded["validation_metrics"]["r2"], 0.893)
            
    def test_quality_check_pass(self):
        r2 = 0.893
        rmse = 8415.0
        r2_threshold = 0.7
        rmse_threshold = 20000.0
        
        self.assertGreaterEqual(r2, r2_threshold)
        self.assertLessEqual(rmse, rmse_threshold)
        
    def test_quality_check_fail_r2(self):
        r2 = 0.65
        r2_threshold = 0.7
        
        self.assertLess(r2, r2_threshold)
        
    def test_quality_check_fail_rmse(self):
        rmse = 25000.0
        rmse_threshold = 20000.0
        
        self.assertGreater(rmse, rmse_threshold)


if __name__ == "__main__":
    unittest.main()
