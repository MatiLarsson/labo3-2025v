# Run model
from multinational.config import ProjectConfig
from multinational.models.lightgbm_regressor import LightGBMModel

if __name__ == "__main__":
    # Load project configuration - automatically finds config file
    config = ProjectConfig.find_and_load()
    
    model = LightGBMModel(config)
    model.prepare_features()
    model.split_data()
    model.optimize()
    model.train_final_ensemble()
    model.predict_for_kaggle()