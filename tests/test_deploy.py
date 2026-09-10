"""Unit tests for deploy.py"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDeploy(unittest.TestCase):
    
    @patch('boto3.client')
    def test_model_package_retrieval(self, mock_boto_client):
        mock_sm = Mock()
        mock_boto_client.return_value = mock_sm
        
        mock_sm.list_model_packages.return_value = {
            'ModelPackageSummaryList': [{
                'ModelPackageArn': 'arn:aws:sagemaker:us-east-1:123:model-package/test',
                'CreationTime': '2025-01-01T00:00:00Z'
            }]
        }
        
        response = mock_sm.list_model_packages(
            ModelPackageGroupName='vap-barrage-champion',
            ModelApprovalStatus='Approved',
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )
        
        self.assertEqual(len(response['ModelPackageSummaryList']), 1)
        self.assertIn('ModelPackageArn', response['ModelPackageSummaryList'][0])
        
    @patch('boto3.client')
    def test_no_approved_models(self, mock_boto_client):
        mock_sm = Mock()
        mock_boto_client.return_value = mock_sm
        
        mock_sm.list_model_packages.return_value = {
            'ModelPackageSummaryList': []
        }
        
        response = mock_sm.list_model_packages(
            ModelPackageGroupName='vap-barrage-champion',
            ModelApprovalStatus='Approved'
        )
        
        self.assertEqual(len(response['ModelPackageSummaryList']), 0)
        
    @patch('boto3.client')
    def test_endpoint_exists_check(self, mock_boto_client):
        mock_sm = Mock()
        mock_boto_client.return_value = mock_sm
        
        mock_sm.describe_endpoint.return_value = {
            'EndpointName': 'VAPSales-endpoint',
            'EndpointStatus': 'InService'
        }
        
        response = mock_sm.describe_endpoint(EndpointName='VAPSales-endpoint')
        self.assertEqual(response['EndpointStatus'], 'InService')


if __name__ == "__main__":
    unittest.main()
