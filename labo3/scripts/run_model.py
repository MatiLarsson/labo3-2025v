from multinational.config import ProjectConfig
from multinational.models.lightgbm_regressor import LightGBMModel

if __name__ == "__main__":
    # Load project configuration
    config_path = "../project_config.yml"
    config = ProjectConfig.from_yaml(config_path=config_path) # Can pass env vars if needed here

    model = LightGBMModel(config)
    model.prepare_features()
    model.split_data()
    model.optimize()
    model.train_final_ensemble()
    model.predict_for_kaggle()

