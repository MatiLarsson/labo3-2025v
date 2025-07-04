# Run dataset
from multinational.config import ProjectConfig
from multinational.dataset_generator import DatasetGenerator

if __name__ == "__main__":
    # Load project configuration - automatically finds config file
    config = ProjectConfig.find_and_load()
    
    generator = DatasetGenerator(config)
    generator.download_data()
    generator.generate_dataset()