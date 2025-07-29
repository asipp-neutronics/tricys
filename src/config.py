import json
import os
import logging
from typing import Dict

# Configure logger
logger = logging.getLogger(__name__)


class ConfigManager:
    """Manage configuration settings from a JSON file."""

    DEFAULT_CONFIG = {
        "package_path": "./example/package.mo",
        "output_dir": "./data"
    }

    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the config manager.

        Args:
            config_path (str): Path to the configuration file. Defaults to 'config.json'.
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, str]:
        """
        Load configuration from the JSON file or return default config.

        Returns:
            Dict[str, str]: Configuration dictionary.
        """
        try:
            if not os.path.exists(self.config_path):
                logger.warning(
                    f"Config file {self.config_path} not found. Using default configuration.")
                return self.DEFAULT_CONFIG

            with open(self.config_path, 'r') as f:
                config = json.load(f)

            # Validate required keys
            for key in self.DEFAULT_CONFIG:
                if key not in config:
                    logger.warning(
                        f"Missing key '{key}' in config. Using default: {self.DEFAULT_CONFIG[key]}")
                    config[key] = self.DEFAULT_CONFIG[key]

            # Validate paths
            if not config["package_path"].endswith('.mo'):
                logger.warning(
                    f"Package path '{config['package_path']}' does not end with '.mo'. May be invalid.")
            if not os.path.exists(os.path.dirname(config["package_path"])):
                logger.warning(
                    f"Package path directory '{os.path.dirname(config['package_path'])}' does not exist.")
            if not os.path.exists(config["output_dir"]):
                logger.info(
                    f"Creating output directory: {config['output_dir']}")
                os.makedirs(config["output_dir"], exist_ok=True)

            logger.info(f"Loaded configuration: {config}")
            return config
        except Exception as e:
            logger.error(
                f"Failed to load config: {str(e)}. Using default configuration.")
            return self.DEFAULT_CONFIG

    def get_package_path(self) -> str:
        """Get the Modelica package path."""
        return self.config["package_path"]

    def get_output_dir(self) -> str:
        """Get the output directory."""
        return self.config["output_dir"]
