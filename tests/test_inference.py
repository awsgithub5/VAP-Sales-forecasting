"""Unit tests for inference.py"""
import unittest
import json
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with patch.dict('sys.modules', {'shap': Mock(), 'xgboost': Mock(), 'lightgbm': Mock()}):
    import inference


class TestInferenceHelpers(unittest.TestCase):
    
    def test_normalize_fips(self):
        self.assertEqual(inference.normalize_fips("1234"), "01234")
        self.assertEqual(inference.normalize_fips("12345"), "12345")
        self.assertEqual(inference.normalize_fips("1234.0"), "01234")
        self.assertEqual(inference.normalize_fips(1234), "01234")
        
    def test_normalize_fips_none(self):
        with self.assertRaises(ValueError):
            inference.normalize_fips(None)
    
    def test_calendar_feats(self):
        date = pd.Timestamp("2025-03-01")
        feats = inference._calendar_feats(date)
        self.assertEqual(feats["calendar_month"], 3)
        self.assertEqual(feats["quarter"], 1)
        self.assertEqual(feats["is_planting_season"], 1)
        self.assertEqual(feats["is_harvest_season"], 0)
        
    def test_get_external_feature_names(self):
        all_features = ["fips_lag_1", "CPI_AllUrban", "county_alltime_share", "TAVG"]
        external = inference.get_external_feature_names(all_features)
        self.assertIn("CPI_AllUrban", external)
        self.assertIn("TAVG", external)
        self.assertNotIn("fips_lag_1", external)
        self.assertNotIn("county_alltime_share", external)


class TestInferenceInputValidation(unittest.TestCase):
    
    def test_input_fn_valid_json(self):
        payload = '{"family": "Barrage", "fips": "04013", "month": 3, "year": 2025}'
        result = inference.input_fn(payload)
        self.assertEqual(result["family"], "Barrage")
        self.assertEqual(result["fips"], "04013")
        
    def test_input_fn_invalid_content_type(self):
        with self.assertRaises(ValueError):
            inference.input_fn("data", content_type="text/plain")


class TestNumpyJSONEncoder(unittest.TestCase):
    
    def test_encode_numpy_float32(self):
        data = {"value": np.float32(123.45)}
        result = json.dumps(data, cls=inference.NumpyJSONEncoder)
        self.assertIn("123.45", result)
        
    def test_encode_numpy_int64(self):
        data = {"value": np.int64(42)}
        result = json.dumps(data, cls=inference.NumpyJSONEncoder)
        self.assertIn("42", result)
        
    def test_encode_numpy_array(self):
        data = {"values": np.array([1, 2, 3])}
        result = json.dumps(data, cls=inference.NumpyJSONEncoder)
        parsed = json.loads(result)
        self.assertEqual(parsed["values"], [1, 2, 3])


class TestModelWrappers(unittest.TestCase):
    
    @patch('inference.xgb.DMatrix')
    def test_native_xgb_wrapper(self, mock_dmatrix):
        mock_booster = Mock()
        mock_booster.predict.return_value = np.array([100.0])
        wrapper = inference._NativeXGBWrapper(mock_booster)
        
        X = pd.DataFrame({"feature1": [1.0]})
        result = wrapper.predict(X)
        
        mock_dmatrix.assert_called_once()
        self.assertEqual(result[0], 100.0)
        
    def test_native_lgb_wrapper(self):
        mock_booster = Mock()
        mock_booster.predict.return_value = np.array([200.0])
        wrapper = inference._NativeLGBWrapper(mock_booster)
        
        X = pd.DataFrame({"feature1": [1.0]})
        result = wrapper.predict(X)
        
        self.assertEqual(result[0], 200.0)


if __name__ == "__main__":
    unittest.main()
