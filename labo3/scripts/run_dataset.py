from multinational.config import ProjectConfig
from multinational.dataset_generator import DatasetGenerator

if __name__ == "__main__":
    # Load project configuration
    config_path = "../project_config.yml"
    config = ProjectConfig.from_yaml(config_path=config_path) # Can pass env vars if needed here

    generator = DatasetGenerator(config)
    generator.download_data()
    generator.generate_dataset()

