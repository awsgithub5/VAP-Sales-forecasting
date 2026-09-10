"""Helper to retrieve VAP configuration from config file"""
import json
import os

def get_config():
    """Load configuration from config.json file"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)
