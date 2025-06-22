import json
import logging
import os

class ConfigManager:
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.config = None
        self.logger = logging.getLogger(__name__)
        self.load_config()
        
    def load_config(self):
        """Load and parse configuration file"""
        try:
            if not os.path.exists(self.config_path):
                self.logger.error(f"Configuration file not found: {self.config_path}")
                raise FileNotFoundError(f"Configuration file {self.config_path} not found")
            
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            self.logger.info(f"Successfully loaded configuration from {self.config_path}")
            self.validate_config()
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Configuration file format error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise
        
    def validate_config(self):
        """Validate configuration parameters"""
        required_sections = ['monitor', 'roi_vertices', 'hough_transform', 'steering', 'logging', 'general']
        
        for section in required_sections:
            if section not in self.config:
                self.logger.warning(f"Missing required configuration section: {section}")
                
        # Add more specific validation if needed
        self.logger.debug("Configuration validation completed")
        
    def get(self, key_path, default=None):
        """Get configuration value using dot notation (e.g., 'steering.deviation_threshold')"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            self.logger.warning(f"Configuration key not found: {key_path}, using default value: {default}")
            return default
        
    def update(self, key_path, value):
        """Update configuration value using dot notation"""
        keys = key_path.split('.')
        config_node = self.config
        
        try:
            for key in keys[:-1]:
                config_node = config_node[key]
            
            config_node[keys[-1]] = value
            self.logger.info(f"Updated configuration: {key_path} = {value}")
            
            # Save updated configuration to file
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            
        except (KeyError, TypeError) as e:
            self.logger.error(f"Failed to update configuration: {str(e)}")
            raise