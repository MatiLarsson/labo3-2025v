# gcs_manager.py
from google.cloud import storage
from google.oauth2 import service_account
from google.api_core import retry
import requests
import os
from loguru import logger

from multinational.config import ProjectConfig


class GCSManager:
    def __init__(self, config: ProjectConfig):
        """
        Initialize GCS Manager.
        
        Args:
            config (ProjectConfig): Project configuration containing GCS settings
        """
        self.gcp = config.gcp
        self._client = None  # Cache the client
    
    @staticmethod
    def is_running_on_gcp():
        """Check if code is running on Google Cloud Platform."""
        try:
            # Try to access metadata server (only available on GCP)
            response = requests.get(
                'http://metadata.google.internal/computeMetadata/v1/project/project-id',
                headers={'Metadata-Flavor': 'Google'},
                timeout=1
            )
            return response.status_code == 200
        except:
            return False
    
    def _get_gcs_client(self):
        """
        Get Google Cloud Storage client with appropriate authentication.
        Detects if running on GCP or local machine.
        
        Returns:
            storage.Client: Authenticated GCS client
        """
        # Return cached client if already created
        if self._client is not None:
            return self._client
            
        # Check if running on Google Cloud (has metadata server)
        if self.is_running_on_gcp():
            logger.info("Detected GCP environment - using default credentials")
            # On GCP, use default service account
            self._client = storage.Client()
        else:
            logger.info("Detected local environment - using service account file")
            # Local machine, use service account file
            credentials_path = os.environ.get(
                'GOOGLE_APPLICATION_CREDENTIALS', 
                None
            )
            
            if not credentials_path or not os.path.exists(credentials_path):
                error_msg = f"Service account file not found: {credentials_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            logger.info(f"Loading credentials from: {credentials_path}")
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self._client = storage.Client(credentials=credentials)
        
        return self._client
    
    def download_file_as_text(self, filename):
        """
        Download a file from GCS as text.
        
        Args:
            filename (str): Name of the file to download
            
        Returns:
            str: File content as text
        """
        file_path = f"{self.gcp['blob_path']}/{filename}"  # Define file_path early
        try:
            client = self._get_gcs_client()
            bucket = client.bucket(self.gcp["bucket_name"])
            blob = bucket.blob(file_path)
            return blob.download_as_text()
        except Exception as e:
            logger.error(f"Failed to download {filename} from {self.gcp['bucket_name']}: {e}")
            raise

    def download_file_as_bytes(self, filename):
        """
        Download a file from GCS as bytes.
        
        Args:
            filename (str): Name of the file to download
            
        Returns:
            bytes: File content as bytes
        """
        file_path = f"{self.gcp['blob_path']}/{filename}"  # Define file_path early
        try:
            client = self._get_gcs_client()
            bucket = client.bucket(self.gcp["bucket_name"])
            blob = bucket.blob(file_path)
            return blob.download_as_bytes()
        except Exception as e:
            logger.error(f"Failed to download {filename} from {self.gcp['bucket_name']}: {e}")
            raise

    def upload_file_from_memory(self, filename, data, content_type='application/octet-stream'):
        """
        Robust upload for large files with massive timeout.
        Use this method for large files that timeout with regular upload_file_from_memory.
        
        Args:
            filename: Name of the file to upload
            data (bytes or io.BytesIO): Data to upload
            content_type (str): MIME type of the content
        """
        file_path = f"{self.gcp['blob_path']}/{filename}"  # Define file_path early
        try:
            client = self._get_gcs_client()
            bucket = client.bucket(self.gcp["bucket_name"])
            blob = bucket.blob(file_path)
            
            # Configure for large files
            blob.chunk_size = 16 * 1024 * 1024  # 16MB chunks for better performance
            
            if hasattr(data, 'seek'):
                data.seek(0)
            
            logger.info(f"Uploading file to gs://{self.gcp['bucket_name']}/{file_path}")

            # Upload with massive timeout and retry - Azure style
            blob.upload_from_file(
                data,
                content_type=content_type,
                timeout=3600,  # 1 hour timeout
                retry=retry.Retry(
                    initial=1.0,
                    maximum=60.0,
                    multiplier=2.0,
                    timeout=3600.0  # 1 hour total retry timeout
                )
            )
            
            logger.info(f"✅ File uploaded successfully to gs://{self.gcp['bucket_name']}/{file_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to upload file to {self.gcp['bucket_name']}/{file_path}: {e}")
            raise

    def get_mlflow_tracking_uri(self):
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

    def verify_mlflow_gcs_access(self, tracking_uri):
        """
        Verify that MLflow server is accessible.
        
        Args:
            tracking_uri (str): MLflow server URI
            
        Returns:
            bool: True if server is accessible
        """
        try:
            import requests
            
            # Test server health endpoint
            health_url = f"{tracking_uri}/health"
            response = requests.get(health_url, timeout=10)
            
            if response.status_code == 200:
                logger.info("✅ MLflow server is healthy and accessible")
                return True
            else:
                logger.warning(f"⚠️ MLflow server returned status: {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ MLflow server check failed: {e}")
            logger.warning("Make sure your MLflow server is running and accessible")
            return False

    def create_mlflow_experiment_if_not_exists(self, experiment_name, tracking_uri):
        """
        Create MLflow experiment if it doesn't exist.
        
        Args:
            experiment_name (str): Name of the experiment
            tracking_uri (str): MLflow server URI
            
        Returns:
            str: Experiment ID
        """
        try:    
            import mlflow
            
            # Set tracking URI to your MLflow server
            mlflow.set_tracking_uri(tracking_uri)
            
            # Try to get existing experiment
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is not None:
                    logger.info(f"✅ Using existing MLflow experiment: {experiment_name}")
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