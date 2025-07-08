import os
import requests
from loguru import logger
import mlflow
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import warnings


def get_mlflow_tracking_uri():
    """
    Get MLflow tracking server URI.
    
    Returns:
        str: MLflow server URI
    """
    try:
        mlflow_server_uri = os.getenv('MLFLOW_TRACKING_URI', None)
    
        return mlflow_server_uri
    except Exception as e:
        logger.error(f"❌ Failed to get MLflow tracking URI: {e}")
        raise ValueError("MLflow tracking URI not set in environment variables")

def verify_mlflow_gcs_access(tracking_uri):
    """
    Verify that MLflow server is accessible.
    
    Args:
        tracking_uri (str): MLflow server URI
        
    Returns:
        bool: True if server is accessible
    """
    try:            
        # Test server health endpoint
        health_url = f"{tracking_uri}/health"
        response = requests.get(health_url, timeout=10)
        
        if response.status_code == 200:
            logger.info("MLflow server is healthy and accessible")
            return True
        else:
            logger.warning(f"⚠️ MLflow server returned status: {response.status_code}")
            return False
            
    except Exception as e:
        logger.warning(f"⚠️ MLflow server check failed: {e}")
        logger.warning("Make sure your MLflow server is running and accessible")
        return False

def create_mlflow_experiment_if_not_exists(experiment_name, tracking_uri):
    """
    Create MLflow experiment if it doesn't exist.
    
    Args:
        experiment_name (str): Name of the experiment
        tracking_uri (str): MLflow server URI
        
    Returns:
        str: Experiment ID
    """
    try:    
        # Set tracking URI to your MLflow server
        mlflow.set_tracking_uri(tracking_uri)
        
        # Try to get existing experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is not None:
                logger.info(f"Using existing MLflow experiment: {experiment_name}")
                return experiment.experiment_id
        except Exception:
            pass  # Experiment doesn't exist
        
        # Create new experiment (artifacts will go to GCS via server config)
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"✅ Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
        
        return experiment_id
        
    except Exception as e:
        logger.error(f"❌ Failed to create MLflow experiment: {e}")
        raise

def fit_predict_sarima_chunk(chunk_data):
    """Fit SARIMA model for a chunk of unique_ids - MUST be at module level for pickling"""
    try:
        chunk_df, chunk_id = chunk_data
        if len(chunk_df) == 0:
            return pd.DataFrame(), chunk_id
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsforecast.arima")
            
            sf_chunk = StatsForecast(
                models=[AutoARIMA(season_length=12)], 
                freq='ME'
            )
            sf_chunk.fit(chunk_df)
            predictions = sf_chunk.predict(h=2)
        
        return predictions, chunk_id
        
    except Exception as e:
        print(f"❌ Error in chunk {chunk_id}: {str(e)}")  # Use print instead of logger
        return pd.DataFrame(), chunk_id