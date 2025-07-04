from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel

class ProjectConfig(BaseModel):
    experiment_name: str
    dataset_generation: Dict[str, Any]  # Matches YAML structure
    model_dataset: Dict[str, Any]  # Added - matches YAML
    strategy: Dict[str, Any]  # Strategy configuration
    gcp: Dict[str, Any]  # Configuration for Google Cloud Storage
    testing_limits: Dict[str, Any]  # Configuration for testing limits
    cv: Dict[str, Any]  # Cross-validation configuration
    optimizer: Dict[str, Any]  # Optuna hyperparameter optimization settings
    final_train: Dict[str, Any]  # Final training configuration

    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from a YAML file."""
        try:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
        except Exception as e:
            raise ValueError(f"Config validation failed: {e}")