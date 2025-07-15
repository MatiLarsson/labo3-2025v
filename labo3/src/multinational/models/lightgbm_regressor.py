#lightgbm_regressor.py
import mlflow
import multinational.models.lightgbm_regressor as lgb
from loguru import logger
import polars as pl
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
import numpy as np
import optuna
import time
import lightgbm as lgb
import tempfile
import os
import shutil
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from multinational.config import ProjectConfig
from multinational.gcs_manager import GCSManager
from multinational.utils import get_mlflow_tracking_uri, verify_mlflow_gcs_access, create_mlflow_experiment_if_not_exists, fit_predict_sarima_chunk


class LightGBMModel:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.experiment_name = self.config.experiment_name
        self.dataset = self.config.dataset
        self.strategy = self.config.strategy
        self.cv = self.config.cv
        self.optimizer = self.config.optimizer
        self.final_train = self.config.final_train
        self.gcp_manager = GCSManager(self.config)
        self.final_model = None
        self.final_params = None
        self.run_id = None
        self._setup_mlflow_tracking()

    def _setup_mlflow_tracking(self):
        """Setup MLflow tracking."""
        logger.info("🔧 Setting up MLflow tracking...")
        
        # Use GCSManager for authentication setup
        tracking_uri = get_mlflow_tracking_uri()
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Verify access
        if not verify_mlflow_gcs_access(tracking_uri):
            raise RuntimeError("Failed to verify MLflow GCS access")
        
        # Create experiment if needed
        create_mlflow_experiment_if_not_exists(
            self.experiment_name, 
            tracking_uri
        )
        
        # Set the experiment for this session
        mlflow.set_experiment(self.experiment_name)
        logger.info("✅ MLflow tracking setup completed")

    def _load_data(self):
        """
        Loads full dataset from storage.
        """
        logger.info("🔄 Loading data from MLflow...")
        
        # Check if dataset exists in MLflow
        existing_runs = mlflow.search_runs(
            experiment_names=[self.experiment_name],
            filter_string="tags.dataset_generated = 'true'",
            max_results=1
        )
        
        if existing_runs.empty:
            raise FileNotFoundError("❌ No dataset found in MLflow. Please generate the dataset first.")
        
        # Download dataset from MLflow storage
        run_id = existing_runs.iloc[0]['run_id']
        client = mlflow.tracking.MlflowClient()
        local_path = client.download_artifacts(run_id, self.dataset["dataset_name"])
        
        self.df = pl.read_parquet(local_path)
        logger.info("✅ Data successfully loaded from storage.")

    def _identify_features(self):
        """Identify numeric and categorical features."""
        logger.info("🔍 Identifying feature types...")
        
        # Numeric features are all columns except categorical and target
        self.num_features = [
            col for col in self.df.columns 
            if col not in self.dataset["cat_features"] and col != self.dataset["target"]
        ]

        self.cat_features = self.dataset["cat_features"]
        
        logger.info(f"Found {len(self.num_features)} numeric features")
        logger.info(f"Found {len(self.cat_features)} categorical features")

    def _clip_extreme_values(self):
        """Clip extreme values in numeric features."""
        logger.info("✂️ Clipping extreme values in dataset...")
        
        clipping_threshold = float(self.dataset["clipping_threshold"])
        
        logger.info("Before clipping:")
        min_vals = self.df.select([pl.col(col).min().alias(f"{col}_min") for col in self.num_features])
        max_vals = self.df.select([pl.col(col).max().alias(f"{col}_max") for col in self.num_features])
        
        overall_min = min_vals.select(pl.min_horizontal(pl.all())).item()
        overall_max = max_vals.select(pl.max_horizontal(pl.all())).item()
        
        logger.info(f"Dataset min: {overall_min:.2e}")
        logger.info(f"Dataset max: {overall_max:.2e}")
        
        # Clip numeric columns
        self.df = self.df.with_columns([
            pl.col(col).clip(-clipping_threshold, clipping_threshold) 
            for col in self.num_features
        ])
        
        logger.info("After clipping:")
        min_vals_after = self.df.select([pl.col(col).min().alias(f"{col}_min") for col in self.num_features])
        max_vals_after = self.df.select([pl.col(col).max().alias(f"{col}_max") for col in self.num_features])
        
        overall_min_after = min_vals_after.select(pl.min_horizontal(pl.all())).item()
        overall_max_after = max_vals_after.select(pl.max_horizontal(pl.all())).item()
        
        logger.info(f"Dataset min: {overall_min_after:.2e}")
        logger.info(f"Dataset max: {overall_max_after:.2e}")
        
        logger.info("✅ Extreme values clipped successfully.")

    def _flag_cherry_rows(self):
        """Cherry-flag rows based on specific criteria."""
        logger.info("🍒 Cherry-flagging rows based on criteria...")
        
        # Flag rows where exists at least self.dataset["positive_quantity_tn_cherry_months"] months (from unbounded preceding up to current month over periodo column) where quantity_tn > 0 (grouping by customer_id, product_id)
        self.df = (
            self.df
            .sort(['customer_id', 'product_id', 'periodo'])  # Ensure proper ordering
            .with_columns(
                pl.col('quantity_tn')
                .gt(0)
                .cast(pl.Int32)  # Convert boolean to int for cum_sum
                .cum_sum()
                .over(['customer_id', 'product_id'])
                .ge(int(self.dataset["positive_quantity_tn_cherry_months"]))  # Convert to boolean flag (True if >= self.dataset["positive_quantity_tn_cherry_months"], False otherwise)
                .cast(pl.Int32)  # Convert boolean to int (1/0)
                .alias('cherry_flag')
            )
        )

        # Log the results
        cherry_flagged_count = self.df.select(pl.col('cherry_flag').sum()).item()
        logger.info("✅ Cherry-flagging completed successfully.")
        logger.info(f"Found {cherry_flagged_count} rows flagged as cherry, out of {len(self.df)} total rows.")

    def _flag_problematic_standardization(self):
        """Flag rows with problematic standardization."""
        logger.info("🚩 Flagging problematic standardization rows...")
        
        # Flag rows with any standardization issues
        self.df = self.df.with_columns(
            (
                pl.col('quantity_tn_cumulative_mean').is_null() | 
                pl.col('quantity_tn_cumulative_mean').is_nan() |
                pl.col('quantity_tn_cumulative_mean').eq(0) |
                pl.col('quantity_tn_cumulative_std').is_null() |
                pl.col('quantity_tn_cumulative_std').is_nan() |
                pl.col('quantity_tn_cumulative_std').eq(0) |
                pl.col('target').is_null() |
                pl.col('target').is_nan()
            ).cast(pl.Int32)  # Convert boolean to int (1/0)
            .alias('invalid_standardization_flag')
        )
        
        # Log the results
        invalid_count = self.df.select(pl.col('invalid_standardization_flag').sum()).item()
        logger.info("✅ Problematic standardization rows flagged.")
        logger.info(f"Found {invalid_count} rows with invalid standardization, out of {len(self.df)} total rows.")

    def _encode_categorical_features(self):
        """Encode categorical features using LabelEncoder."""
        logger.info("🏷️ Encoding categorical features...")
        
        encoded_columns = []
        
        for col in self.cat_features:
            # Create and fit label encoder
            le = LabelEncoder()
            col_values = self.df[col].to_numpy().astype(str)
            col_values = [str(val) if val is not None else 'missing' for val in col_values]
            encoded_values = le.fit_transform(col_values)
            
            encoded_columns.append(pl.Series(col, encoded_values))
        
        if encoded_columns:
            self.df = self.df.with_columns(encoded_columns)
        
        logger.info("✅ Categorical features encoded successfully.")

    def _add_extra_features(self):
        if eval(str(self.dataset["add_12m_sarima_features"])):
            if int(self.dataset["max_z_lag_periods"]) < 11:
                logger.warning("❗ Cannot add 12 month SARIMA features because max_z_lag_periods is less than 11. Skipping...")
            else:
                logger.info("➕ Adding 12m SARIMA predictions to the dataset (for cherry & non z problematic rows)..")

                try:
                    lag_columns = [col for col in self.df.columns if col.startswith('quantity_tn_lag_') and int(col.split('_')[-1]) <= 11]

                    # Add row index to the main dataframe ONCE
                    self.df = self.df.with_row_index("original_row_id")

                    # Filter and prepare data for SARIMA
                    sarima_df = (
                        self.df
                        .filter(pl.col('cherry_flag') == 1)
                        .filter(pl.col('invalid_standardization_flag') == 0)
                        .select(["original_row_id"] + ["periodo"] + lag_columns + ["quantity_tn"])
                        .rename({
                            'quantity_tn': 'lag_0',
                            **{f'quantity_tn_lag_{i}': f'lag_{i}' for i in range(1, 12)}
                        })
                        .with_columns([
                            pl.col(f"lag_{i}").fill_null(0.0) for i in range(12)
                        ])
                    )

                    # Update the melt operation to include original_row_id
                    forecast_df = (
                        sarima_df
                        .melt(
                            id_vars=["original_row_id", "periodo"],
                            value_vars=[f"lag_{i}" for i in range(12)],
                            variable_name="lag",
                            value_name="y"
                        )
                        .with_columns([
                            pl.col("lag").str.extract(r"lag_(\d+)").cast(pl.Int32).alias("lag_num"),
                            (pl.col("periodo") // 100).alias("year"),
                            (pl.col("periodo") % 100).alias("month")
                        ])
                        .with_columns([
                            (pl.col("month") - pl.col("lag_num")).alias("target_month_raw"),
                            pl.col("year").alias("target_year")
                        ])
                        .with_columns([
                            pl.when(pl.col("target_month_raw") <= 0)
                            .then(((pl.col("target_month_raw") - 1) % 12) + 1)
                            .otherwise(pl.col("target_month_raw"))
                            .alias("target_month"),
                            
                            pl.when(pl.col("target_month_raw") <= 0)
                            .then(pl.col("target_year") - ((-pl.col("target_month_raw")) // 12 + 1))
                            .otherwise(pl.col("target_year"))
                            .alias("target_year_final")
                        ])
                        .with_columns([
                            pl.date(
                                pl.col("target_year_final"),
                                pl.col("target_month"),
                                1
                            )
                            .dt.month_end()
                            .alias("ds"),
                            ("row_" + pl.col("original_row_id").cast(pl.Utf8)).alias("unique_id")
                        ])
                        .select(["unique_id", "ds", "y", "original_row_id", "lag_num"])
                        .sort(["original_row_id", "lag_num"])
                        .with_columns([
                            pl.col("y").fill_null(0.0).over("unique_id")
                        ])
                    )

                    # Convert to pandas only for StatsForecast
                    forecast_df_pandas = forecast_df.select(["unique_id", "ds", "y"]).to_pandas()
                    forecast_df_pandas['ds'] = pd.to_datetime(forecast_df_pandas['ds'])

                    logger.info(f"🚀 Starting optimized parallel SARIMA fitting...")

                    # Get unique IDs using Polars
                    unique_ids_pl = forecast_df.select("unique_id").unique().sort("unique_id")
                    unique_ids = unique_ids_pl.to_pandas()['unique_id'].values

                    n_chunks = min(os.cpu_count(), len(unique_ids))
                    chunk_size = max(1, len(unique_ids) // n_chunks)
                    
                    # Prepare data for parallel processing (pure data, no class references)
                    chunks = []
                    for i in range(n_chunks):
                        start_idx = i * chunk_size
                        end_idx = min((i + 1) * chunk_size, len(unique_ids))
                        
                        if start_idx < len(unique_ids):
                            chunk_unique_ids = unique_ids[start_idx:end_idx].tolist()
                            
                            # Use Polars filtering (orders of magnitude faster than pandas)
                            chunk_df_pl = forecast_df.filter(
                                pl.col("unique_id").is_in(chunk_unique_ids)
                            )
                            
                            # Convert to pandas only for this specific chunk
                            chunk_df = (
                                chunk_df_pl
                                .select(["unique_id", "ds", "y"])
                                .to_pandas()
                            )
                            chunk_df['ds'] = pd.to_datetime(chunk_df['ds'])
                            
                            chunks.append((chunk_df, i))
                            logger.info(f"✅ Prepared chunk {i}: {len(chunk_df)} rows, {len(chunk_unique_ids)} series")

                    logger.info(f"🎯 All {len(chunks)} chunks prepared")
                    logger.info(f"📊 Processing {len(unique_ids)} time series in {len(chunks)} parallel chunks")

                    # Parallel processing with the external function
                    predictions_list = []
                    with ProcessPoolExecutor(max_workers=n_chunks) as executor:
                        # Submit all jobs using the module-level function
                        future_to_chunk = {
                            executor.submit(fit_predict_sarima_chunk, chunk): i 
                            for i, chunk in enumerate(chunks)
                        }
                        
                        # Collect results
                        for future in as_completed(future_to_chunk):
                            chunk_id = future_to_chunk[future]
                            try:
                                result, returned_chunk_id = future.result()
                                if len(result) > 0:
                                    predictions_list.append(result)
                                    logger.info(f"✅ Completed chunk {returned_chunk_id}")
                            except Exception as e:
                                logger.error(f"❌ Chunk {chunk_id} failed: {str(e)}")
                                raise e

                    # Combine predictions and continue with your existing code
                    if predictions_list:
                        predictions = pd.concat(predictions_list, ignore_index=True)
                        logger.info(f"✅ Combined {len(predictions)} predictions from {len(predictions_list)} chunks")
                    else:
                        logger.error("❌ No predictions generated")
                        raise RuntimeError("No predictions generated from SARIMA fitting")

                    # Rest of your existing code for joining back to self.df
                    predictions_pl = pl.from_pandas(predictions).select(["unique_id", "AutoARIMA"])

                    # Update the final mapping to use original_row_id
                    predictions_mapped = (
                        predictions_pl
                        .with_columns(
                            pl.col("unique_id").str.extract(r"row_(\d+)").cast(pl.Int32).alias("original_row_id")
                        )
                        .select(["original_row_id", "AutoARIMA"])
                        .rename({"AutoARIMA": "quantity_tn_predicted_sarima_12m"})
                    )

                    # Join back using original_row_id (NO second with_row_index call)
                    self.df = (
                        self.df
                        .join(predictions_mapped, on="original_row_id", how="left")
                        .drop("original_row_id")  # Clean up the temporary column
                        .with_columns([
                            pl.col("quantity_tn_predicted_sarima_12m").cast(pl.Float64)
                        ])
                        .with_columns([
                            pl.when(
                                pl.col("quantity_tn_cumulative_mean").is_not_nan() & 
                                pl.col("quantity_tn_cumulative_std").is_not_nan() & 
                                pl.col("quantity_tn_standardized").is_not_nan() &
                                pl.col("quantity_tn_predicted_sarima_12m").is_not_nan()
                            ).then(
                                ((pl.col("quantity_tn_predicted_sarima_12m") - pl.col("quantity_tn_cumulative_mean")) / pl.col("quantity_tn_cumulative_std"))
                                - pl.col("quantity_tn_standardized")
                            ).otherwise(pl.lit(None))
                            .alias("quantity_target_sarima_12m_prediction")
                        ])
                    )

                    logger.info("✅ 12 month SARIMA predictions added successfully using parallel processing.")
                    
                except Exception as e:
                    logger.error(f"❌ Error adding SARIMA features: {str(e)}")
                    raise e
        
    def prepare_features(self):
        """Complete feature preparation pipeline."""
        with mlflow.start_run(run_name="features_preparation"):
            # Check if a dataset already exists for this experiment in MLflow, if so, skip generation
            existing_runs = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                filter_string="tags.features_prepared_dataset_generated = 'true'",
                max_results=1
            )

            if not existing_runs.empty:
                logger.info("📂 Features prepared dataset already generated for this experiment. Skipping.")

                # Download features prepared dataset from MLflow storage
                run_id = existing_runs.iloc[0]['run_id']
                client = mlflow.tracking.MlflowClient()

                logger.info(f"Downloading features prepared dataset from MLflow run {run_id}...")

                local_path = client.download_artifacts(run_id, f"features_prepared_{self.dataset["dataset_name"]}")
                
                self.df = pl.read_parquet(local_path)
                logger.info("✅ Features prepared dataset successfully loaded from storage.")

                self._identify_features()

                return
            
            logger.info("🔄 Starting feature preparation pipeline...")

            self._load_data()
            self._identify_features() # Needed for _clip_extreme_values
            self._clip_extreme_values()
            self._flag_cherry_rows()
            self._flag_problematic_standardization()
            self._encode_categorical_features()
            self._add_extra_features()
            self._identify_features() # Need to reset due to added extra features

            # Log dataset with prepared features to MLflow
            temp_path = "/tmp/features_prepared_dataset.parquet"
            
            self.df.write_parquet(temp_path)
            
            logger.info(f"Logging features prepared dataset to MLflow at features_prepared_{self.dataset["dataset_name"]}...")

            mlflow.log_artifact(
                temp_path,
                artifact_path=f"features_prepared_{self.dataset["dataset_name"]}"
            )

            os.remove(temp_path)

            logger.info(f"Dataset generated and logged to MLflow at {self.dataset['dataset_name']}")
            
            # Log dataset metadata
            mlflow.log_param("features_prepared_dataset_len", str(self.df.shape[0]))
            mlflow.log_param("features_prepared_dataset_columns", str(self.df.shape[1]))

            # Tag the dataset as generated
            mlflow.set_tag("features_prepared_dataset_generated", "true")

            logger.info("✅ Feature preparation pipeline completed.")

    def _create_time_based_folds(self, train_dataset):
        """Create time-based folds for cross-validation."""
        month_counts = (train_dataset
                    .group_by(self.dataset["period"])
                    .agg(pl.len().alias('count'))
                    .sort(self.dataset["period"]))
        
        total_samples = month_counts['count'].sum()
        target_samples_per_fold = total_samples / self.cv["n_folds"]
        
        months = month_counts[self.dataset["period"]].to_list()
        counts = month_counts['count'].to_list()
        
        folds = []
        current_fold_months = []
        current_fold_samples = 0
        
        for i, (month, count) in enumerate(zip(months, counts)):
            current_fold_months.append(month)
            current_fold_samples += count

            # Get the next month's count for the comparison
            next_month_count = counts[i + 1] if i + 1 < len(counts) else 0
            
            # If we've reached the target size or this is the last month
            if (current_fold_samples >= target_samples_per_fold - next_month_count/2 or
                i == len(months) - 1):
                
                folds.append({
                    'months': current_fold_months.copy(),
                    'samples': current_fold_samples,
                    'start_month': current_fold_months[0],
                    'end_month': current_fold_months[-1]
                })
                
                current_fold_months = []
                current_fold_samples = 0
                
                # Stop if we have enough folds
                if len(folds) == self.cv["n_folds"]:
                    break

        self.folds = folds
        logger.info(f"Created {len(folds)} time-based folds.")
        for fold in self.folds:
            logger.info(f"Fold from {fold['start_month']} to {fold['end_month']} with {fold['samples']} samples.")

    def _get_fold_indices(self, train_dataset):
        """Convert time-based folds to row indices"""
        fold_indices = []
        
        for fold in self.folds:
            fold_mask = train_dataset.select(pl.col(self.dataset["period"]).is_in(fold['months']))
            fold_idx = np.where(fold_mask.to_numpy().flatten())[0]
            fold_indices.append(fold_idx)
        
        self.fold_indices = fold_indices

    def split_data(self):
        """Split the dataset into training and testing sets."""
        logger.info("🔄 Splitting data...")
        
        # Filter datasets based on the defined periods, the cherry flag and the problematic standardization flag
        train_dataset = self.df.filter(pl.col(self.dataset["period"]) < self.strategy["test_month"]).filter(pl.col('cherry_flag') == 1).filter(pl.col('invalid_standardization_flag') == 0)
        final_train_dataset = self.df.filter(pl.col(self.dataset["period"]) <= self.strategy["test_month"]).filter(pl.col('cherry_flag') == 1).filter(pl.col('invalid_standardization_flag') == 0)
        test_dataset = self.df.filter(pl.col(self.dataset["period"]) == self.strategy["test_month"]).filter(pl.col('cherry_flag') == 1).filter(pl.col('invalid_standardization_flag') == 0)
        self.kaggle_dataset = self.df.filter(pl.col(self.dataset["period"]) == self.strategy["kaggle_month"])

        # Calculate target number of rows for final_train_dataset
        target_rows = len(train_dataset) // self.cv["n_folds"] * (self.cv["n_folds"] - 1)

        # Trim final_train_dataset by removing oldest periods
        if len(final_train_dataset) > target_rows:
            # Sort by period descending and take the newest rows
            final_train_dataset = (
                final_train_dataset
                .sort(self.dataset["period"], descending=True)
                .head(target_rows)
            )

        # Store datasets info for MLflow logging
        self.train_size = len(train_dataset)
        self.final_train_size = len(final_train_dataset)
        self.test_size = len(test_dataset)
        self.kaggle_size = len(self.kaggle_dataset)

        logger.info(f"Training dataset size: {self.train_size}")
        logger.info(f"Final training dataset size: {self.final_train_size}")
        logger.info(f"Testing dataset size: {self.test_size}")
        logger.info(f"Kaggle dataset size: {self.kaggle_size}")

        # Gather global series needed for training
        self.global_quantity_tn_standardized_train = train_dataset.select(pl.col('quantity_tn_standardized')).to_numpy().flatten()
        self.global_quantity_tn_standardized_test = test_dataset.select(pl.col('quantity_tn_standardized')).to_numpy().flatten()
        self.global_mean_values_train = train_dataset.select(pl.col('quantity_tn_cumulative_mean')).to_numpy().flatten()
        self.global_mean_values_test = test_dataset.select(pl.col('quantity_tn_cumulative_mean')).to_numpy().flatten()
        self.global_std_values_train = train_dataset.select(pl.col('quantity_tn_cumulative_std')).to_numpy().flatten()
        self.global_std_values_test = test_dataset.select(pl.col('quantity_tn_cumulative_std')).to_numpy().flatten()

        self.categorical_columns_idx = [final_train_dataset.drop('target').columns.index(col) for col in self.dataset["cat_features"]]

        # Calculate sample weights for taining dataset
        weight_lookup_train = (
            train_dataset
            .group_by(['customer_id', 'product_id'])
            .agg(pl.col('quantity_tn').mean().alias('weight'))
        )
        self.sample_weight_train = (
            train_dataset
            .join(weight_lookup_train, on=['customer_id', 'product_id'], how='left')
            .select('weight')
            .to_numpy()
            .flatten()
        )

        # Calculate sample weights for final training dataset
        weight_lookup_final_train = (
            final_train_dataset
            .group_by(['customer_id', 'product_id'])
            .agg(pl.col('quantity_tn').mean().alias('weight'))
        )
        self.sample_weight_final_train = (
            final_train_dataset
            .join(weight_lookup_final_train, on=['customer_id', 'product_id'], how='left')
            .select('weight')
            .to_numpy()
            .flatten()
        )

        self.X_train = train_dataset.drop([self.dataset["target"]]).to_numpy()
        self.X_test = test_dataset.drop([self.dataset["target"]]).to_numpy()
        self.X_final_train = final_train_dataset.drop([self.dataset["target"]]).to_numpy()
        self.y_train = train_dataset.select(pl.col(self.dataset["target"])).to_numpy().flatten()
        self.y_test = test_dataset.select(pl.col(self.dataset["target"])).to_numpy().flatten()
        self.y_final_train = final_train_dataset.select(pl.col(self.dataset["target"])).to_numpy().flatten()

        self._create_time_based_folds(train_dataset)
        self._get_fold_indices(train_dataset)

        logger.info("✅ Data splitted successfully.")

    def _setup_optuna_storage(self):
        """Setup Optuna storage with GCS backup."""
        # Local SQLite path
        local_db_path = f"/tmp/optuna_study_{self.experiment_name}.db"
        gcs_db_path = f"optuna_studies/optuna_study_{self.experiment_name}.db"
        
        # Try to download existing study from GCS
        try:
            logger.info("🔄 Checking for existing Optuna study in GCS...")
            db_content = self.gcp_manager.download_file_as_bytes(gcs_db_path)
            with open(local_db_path, 'wb') as f:
                f.write(db_content)
            logger.info("✅ Downloaded existing Optuna study from GCS")
        except Exception as e:
            logger.info(f"🆕 No existing study found in GCS (or error: {e}), starting fresh")
        
        return f"sqlite:///{local_db_path}", local_db_path, gcs_db_path
    
    def _backup_study_to_gcs(self, local_db_path, gcs_db_path):
        """Backup Optuna study to GCS after each trial."""
        try:
            with open(local_db_path, 'rb') as f:
                # Use the existing upload_file_from_memory method
                self.gcp_manager.upload_file_from_memory(
                    filename=gcs_db_path,
                    data=f,
                    content_type='application/octet-stream'
                )
            logger.debug("💾 Study backed up to GCS")
        except Exception as e:
            logger.warning(f"⚠️ Failed to backup study to GCS: {e}")

    def _get_seeds(self):
        """Generate consistent seeds for final models."""
        num_seeds = int(self.final_train["num_seeds"])
        np.random.seed(42)  # Always use the same seed
        seeds = np.random.randint(1, 10000, size=num_seeds).tolist()
        return seeds, num_seeds

    def _ensure_final_models_in_class(self):
        logger.info("🔄 Ensuring final models are loaded into the class...")
        if not hasattr(self, 'final_models') or not self.final_models or len(self.final_models) < int(self.final_train["num_seeds"]):
            logger.info("🔄 Loading final models from MLflow...")

            self.final_models = []

            seeds, num_seeds = self._get_seeds()

            for i, seed in enumerate(seeds):
                # Check if model for this seed already exists in the experiment
                existing_model_runs = mlflow.search_runs(
                    experiment_names=[self.experiment_name],
                    filter_string=f"tags.final_model_seed = '{str(seed)}'",
                    max_results=1
                )
                
                if not existing_model_runs.empty:
                    # Load existing model
                    run_id = existing_model_runs.iloc[0]['run_id']
                    client = mlflow.tracking.MlflowClient()

                    # Download and load the model
                    model_path = client.download_artifacts(run_id, f"final_models/final_model_{i+1}.txt")
                    model = lgb.Booster(model_file=model_path)
                    logger.info(f"Loaded existing model {i+1}/{num_seeds} with seed: {seed}")
                    self.final_models.append(model)
                    logger.info(f"✅ Model {i+1} loaded successfully")
                else:
                    logger.error(f"❌ No existing model found for seed {seed}. Please ensure the final model is trained and logged correctly.")
                    raise RuntimeError("No existing model found for seed. Please ensure the final model is trained and logged correctly.")
    
    def optimize(self):
        """
        Optimize the model's params with GCS-backed Optuna storage for recovery.
        """
        # Custom metric function for LightGBM
        def total_forecast_error_cv(y_pred, y_true): # Needs (preds: numpy 1-D array, eval_data: Dataset)
            """
            Custom metric for LightGBM - Total Forecast Error - Uses current fold indices
            Returns: eval_name, eval_result, is_higher_better
            """
            y_true_array = y_true.get_label()

            tfe = pl.DataFrame({
                'y_true': y_true_array,
                'y_pred': y_pred,
                'quantity_tn_standardized': self.current_val_fold_quantity_tn_standardized,
                'mean': self.current_val_fold_mean,
                'std': self.current_val_fold_std
            }).with_columns([
                # Reverse transform predictions
                (((pl.col('quantity_tn_standardized') + pl.col('y_pred')) * pl.col('std')) + pl.col('mean')).alias('quantity_tn_pred'),
                # Reverse transform true values
                (((pl.col('quantity_tn_standardized') + pl.col('y_true')) * pl.col('std')) + pl.col('mean')).alias('quantity_tn_true')
            ]).with_columns([
                # Zero out negative predictions
                pl.col('quantity_tn_pred').clip(0.0, None).alias('quantity_tn_pred')
            ]).with_columns([
                # Calculate absolute differences
                (pl.col('quantity_tn_true') - pl.col('quantity_tn_pred')).abs().alias('abs_diff')
            ]).select([
                pl.col('abs_diff').sum().alias('numerator'),
                pl.col('quantity_tn_true').sum().alias('denominator')
            ]).with_columns([
                (pl.col('numerator') / (pl.col('denominator'))).alias('tfe')
            ])['tfe'].item()
            
            return 'total_forecast_error', tfe, False  # return (eval_name: str, eval_result: float, is_higher_better: bool)

        def objective(trial):
            """
            Optuna objective function with MLflow checkpointing
            """

            params = {
                'objective': self.optimizer["base_model_params"]["objective"],
                'num_iterations': self.optimizer["base_model_params"]["num_iterations"], # Big enough to let early stopping work
                'learning_rate': trial.suggest_float('learning_rate', self.optimizer["param_ranges"]["learning_rate"][0],self.optimizer["param_ranges"]["learning_rate"][1], log=True),
                'num_leaves': trial.suggest_int('num_leaves', self.optimizer["param_ranges"]["num_leaves"][0], self.optimizer["param_ranges"]["num_leaves"][1]),
                'seed': self.optimizer["base_model_params"]["seed"],
                'linear_tree': self.optimizer["base_model_params"]["linear_tree"],
                'max_depth': trial.suggest_int('max_depth', self.optimizer["param_ranges"]["max_depth"][0], self.optimizer["param_ranges"]["max_depth"][1]),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', self.optimizer["param_ranges"]["min_data_in_leaf"][0], self.optimizer["param_ranges"]["min_data_in_leaf"][1]),
                'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', self.optimizer["param_ranges"]["min_sum_hessian_in_leaf"][0], self.optimizer["param_ranges"]["min_sum_hessian_in_leaf"][1], log=True),
                'bagging_fraction': trial.suggest_float('bagging_fraction', self.optimizer["param_ranges"]["bagging_fraction"][0], self.optimizer["param_ranges"]["bagging_fraction"][1]),
                'bagging_freq': trial.suggest_int('bagging_freq', self.optimizer["param_ranges"]["bagging_freq"][0], self.optimizer["param_ranges"]["bagging_freq"][1]),
                'bagging_seed': self.optimizer["base_model_params"]["bagging_seed"],
                'feature_fraction': trial.suggest_float('feature_fraction', self.optimizer["param_ranges"]["feature_fraction"][0], self.optimizer["param_ranges"]["feature_fraction"][1]),
                'feature_fraction_seed': self.optimizer["base_model_params"]["feature_fraction_seed"],
                'lambda_l1': trial.suggest_float('lambda_l1', self.optimizer["param_ranges"]["lambda_l1"][0], self.optimizer["param_ranges"]["lambda_l1"][1], log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', self.optimizer["param_ranges"]["lambda_l2"][0], self.optimizer["param_ranges"]["lambda_l2"][1], log=True),
                'linear_lambda': trial.suggest_float('linear_lambda', self.optimizer["param_ranges"]["linear_lambda"][0], self.optimizer["param_ranges"]["linear_lambda"][1], log=True),
                'verbosity': self.optimizer["base_model_params"]["verbosity"], # < 0: Fatal, = 0: Error (Warning), = 1: Info, > 1: Debug
                'max_bin': trial.suggest_int('max_bin', self.optimizer["param_ranges"]["max_bin"][0], self.optimizer["param_ranges"]["max_bin"][1]), # max number of bins that feature values will be bucketed in
                'metric': self.optimizer["base_model_params"]["metric"],
            }

            max_early_stopping_rounds = 9999

            # Calculate early stopping rounds based on learning rate
            if 'learning_rate' in params and params['learning_rate'] > 0:
                current_round_early_stopping = min(int(400 + 4 / params['learning_rate']), max_early_stopping_rounds)
            else:
                current_round_early_stopping = 400

            cv_scores = []
            fold_best_iterations = []

            with mlflow.start_run(nested=True, run_name=f"optuna_trial_{trial.number}"):
                mlflow.log_params(params)
                mlflow.log_param("early_stopping_rounds", current_round_early_stopping)

                # Log trial parameters to MLflow
                for fold_num in range(len(self.folds)):
                    # Create train/val split for this fold
                    val_idx = self.fold_indices[fold_num]
                    train_idx = np.concatenate([self.fold_indices[i] for i in range(len(self.folds)) if i != fold_num])
                    
                    # Create fold datasets
                    fold_train = lgb.Dataset(
                        data=self.X_train[train_idx],
                        label=self.y_train[train_idx],
                        weight=self.sample_weight_train[train_idx],
                        categorical_feature=self.categorical_columns_idx
                    )
                    
                    fold_val = lgb.Dataset(
                        data=self.X_train[val_idx],
                        label=self.y_train[val_idx],
                        weight=self.sample_weight_train[val_idx],
                        categorical_feature=self.categorical_columns_idx,
                        reference=fold_train
                    )
                    
                    # Store indices for custom metric
                    self.current_val_fold_quantity_tn_standardized = self.global_quantity_tn_standardized_train[val_idx]
                    self.current_val_fold_mean = self.global_mean_values_train[val_idx]
                    self.current_val_fold_std = self.global_std_values_train[val_idx]
                    
                    # Train model
                    model = lgb.train(
                        params,
                        fold_train,
                        valid_sets=[fold_val],
                        valid_names=['valid'],
                        feval=total_forecast_error_cv,
                        callbacks=[lgb.early_stopping(stopping_rounds=current_round_early_stopping)]
                    )
                    
                    cv_scores.append(model.best_score['valid']['total_forecast_error'])
                    fold_best_iterations.append(model.best_iteration)

                mean_cv_score = np.mean(cv_scores)
                
                mlflow.log_metric("cv_mean_tfe", mean_cv_score)
                mlflow.log_metric("p75_iterations", np.percentile(fold_best_iterations, 75))
                
                # Log individual fold scores
                for i, score in enumerate(cv_scores):
                    mlflow.log_metric(f"fold_{i}_tfe", score)
                    mlflow.log_metric(f"fold_{i}_iterations", fold_best_iterations[i])

            trial.set_user_attr('fold_scores', cv_scores)
            trial.set_user_attr('fold_iterations', fold_best_iterations)
            trial.set_user_attr('percentile_75_iterations', np.percentile(fold_best_iterations, 75))

            self._backup_study_to_gcs(local_db_path, gcs_db_path)

            return mean_cv_score
        
        # Setup storage
        storage_url, local_db_path, gcs_db_path = self._setup_optuna_storage()

        with mlflow.start_run(run_name="lightgbm_optimization") as parent_run:
            self.run_id = parent_run.info.run_id

            # Log configuration
            mlflow.log_params({
                "experiment_name": self.experiment_name,
                "train_size": self.train_size,
                "test_size": self.test_size,
                "final_train_size": self.final_train_size,
                "kaggle_size": self.kaggle_size,
                "test_month": self.strategy["test_month"],
                "n_folds": self.cv["n_folds"],
                "n_trials": self.optimizer["n_trials"]
            })
            mlflow.log_dict(self.folds, "cv_folds.json")

            study = optuna.create_study(
                study_name=self.optimizer["study_name"],
                direction=self.optimizer["direction"],
                storage=storage_url,
                sampler=optuna.samplers.TPESampler(seed=42),  # Bayesian optimization with TPE
                load_if_exists=True  # Load existing study if it already exists
            )

            # Check how many trials are already completed
            completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            remaining_trials = max(0, self.optimizer["n_trials"] - completed_trials)

            if completed_trials > 0:
                logger.info(f"🔄 Found {completed_trials} completed trials, running {remaining_trials} more...")
            
            if remaining_trials > 0:
                logger.info("🔄 Starting Optuna optimization with GCS-backed storage...")
                study.optimize(objective, n_trials=remaining_trials)
                time.sleep(2)  # Give some time for the logger to flush
            else:
                logger.info("✅ All trials already completed")

            # Final backup
            self._backup_study_to_gcs(local_db_path, gcs_db_path)

            # Access best trial information
            best_trial = study.best_trial

            logger.info(f"Best Optuna trial: {best_trial.number}")
            logger.info(f"Best CV AVG tfe: {best_trial.value}")
            logger.info(f"Best parameters: {best_trial.params}")

            mlflow.log_param("best_trial_number", best_trial.number)
            mlflow.log_metric("best_cv_tfe", best_trial.value)
            mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})

            final_num_iterations = int(best_trial.user_attrs['percentile_75_iterations'])

            logger.info(f"Final model will use P75 of best iterations: {final_num_iterations}")

            # Prepare final parameters - COMPLETE parameter set used during validation
            self.final_params = {
                'objective': 'regression',
                'num_iterations': final_num_iterations,
                'learning_rate': best_trial.params['learning_rate'],
                'num_leaves': best_trial.params['num_leaves'],
                'seed': 42,
                'linear_tree': True,
                'max_depth': best_trial.params['max_depth'],
                'min_data_in_leaf': best_trial.params['min_data_in_leaf'],
                'min_sum_hessian_in_leaf': best_trial.params['min_sum_hessian_in_leaf'],
                'bagging_fraction': best_trial.params['bagging_fraction'],
                'bagging_freq': best_trial.params['bagging_freq'],
                'bagging_seed': 42,
                'feature_fraction': best_trial.params['feature_fraction'],
                'feature_fraction_seed': 42,
                'lambda_l1': best_trial.params['lambda_l1'],
                'lambda_l2': best_trial.params['lambda_l2'],
                'linear_lambda': best_trial.params['linear_lambda'],
                'verbosity': -1,
                'max_bin': best_trial.params['max_bin']
            }

            # Log final parameters if not already logged
            mlflow.log_params({f"final_{k}": v for k, v in self.final_params.items()})

            logger.info(f"Final parameters: {self.final_params}")

    def train_final_ensemble(self):
        """Train the final model with optimized parameters and log to MLflow."""
        if self.final_params is None:
            raise ValueError("Must run optimize() before training final model")
            
        # Check if final training already completed
        existing_runs = mlflow.search_runs(
            experiment_names=[self.experiment_name],
            filter_string="tags.final_training_completed = 'true'",
            max_results=1
        )
        
        if not existing_runs.empty:
            logger.info("✅ Final training already completed, skipping")
            self._ensure_final_models_in_class()
            logger.info("Final models loaded successfully.")
            return
            
        logger.info("🚀 Starting final models training with optimized parameters...")
        
        with mlflow.start_run(run_name="final_lightgbm_models", nested=True):
            # Create full training dataset
            lgb_final_train_dataset = lgb.Dataset(
                data=self.X_final_train,
                label=self.y_final_train,
                weight=self.sample_weight_final_train,
                categorical_feature=self.categorical_columns_idx
            )
            
            seeds, num_seeds = self._get_seeds()

            self.final_models = []
            self.aggregated_feature_importance = {}

            for i, seed in enumerate(seeds):
                # Check if model for this seed already exists in the experiment
                existing_model_runs = mlflow.search_runs(
                    experiment_names=[self.experiment_name],
                    filter_string=f"tags.final_model_seed = '{str(seed)}'",
                    max_results=1
                )
                
                if not existing_model_runs.empty:
                    # Load existing model
                    run_id = existing_model_runs.iloc[0]['run_id']
                    client = mlflow.tracking.MlflowClient()
                    
                    try:
                        # Download and load the model
                        model_path = client.download_artifacts(run_id, f"final_models/final_model_{i+1}.txt")
                        model = lgb.Booster(model_file=model_path)
                        logger.info(f"Loaded existing model {i+1}/{num_seeds} with seed: {seed}")
                        self.final_models.append(model)
                        
                        # Try to load feature importance for aggregation
                        try:
                            importance_path = client.download_artifacts(run_id, f"feature_importance_{i+1}.json")
                            with open(importance_path, 'r') as f:
                                import json
                                importance_dict = json.load(f)
                            
                            # Aggregate feature importance
                            for feature, importance in importance_dict.items():
                                if feature not in self.aggregated_feature_importance:
                                    self.aggregated_feature_importance[feature] = 0.0
                                self.aggregated_feature_importance[feature] += float(importance)
                            
                            logger.info(f"Loaded existing feature importance for model {i+1}/{num_seeds} with seed: {seed}")
                            continue
                            
                        except Exception as e:
                            # Model exists but feature importance doesn't - regenerate it
                            logger.warning(f"Model {i+1}/{num_seeds} with {seed} exists but feature importance missing: {e}")
                            logger.info(f"Regenerating feature importance for model {i+1}/{num_seeds} with {seed}")
                            
                            # Get feature importance from loaded model
                            model_feature_importance = model.feature_importance(importance_type='gain')
                            feature_names = self.df.drop([self.dataset["target"]]).columns

                            importance_dict = {
                                feature: float(importance)
                                for feature, importance in zip(feature_names, model_feature_importance)
                            }

                            sorted_importance_dict = dict(sorted(importance_dict.items(), 
                                key=lambda x: x[1], 
                                reverse=True)
                            )

                            # Log the missing feature importance to the existing run
                            with mlflow.start_run(run_id=run_id):
                                mlflow.log_dict(sorted_importance_dict, f"feature_importance_{i+1}.json")

                            # Aggregate feature importance
                            for feature, importance in importance_dict.items():
                                if feature not in self.aggregated_feature_importance:
                                    self.aggregated_feature_importance[feature] = 0.0
                                self.aggregated_feature_importance[feature] += importance
                            
                            logger.info(f"Loaded existing model {i+1}/{num_seeds} with seed: {seed} and regenerated it's feature importance")
                            continue
                        
                    except Exception as e:
                        logger.warning(f"Failed to load existing model {i+1}/{num_seeds} for seed {seed}: {e}, retraining...")

                logger.info(f"Training final model {i+1}/{num_seeds} with seed: {seed}")
                
                # Create a copy of params for this seed
                final_train_params = self.final_params.copy()
                final_train_params['seed'] = seed
                final_train_params['bagging_seed'] = seed
                final_train_params['feature_fraction_seed'] = seed

                # Train final model
                model = lgb.train(
                    params=final_train_params,
                    train_set=lgb_final_train_dataset
                )
            
                self.final_models.append(model)

                model_dir = tempfile.mkdtemp()
                model_path = os.path.join(model_dir, f"final_model_{i+1}.txt")
                model.save_model(model_path)
                
                # Log artifact WITHOUT tags parameter
                mlflow.log_artifact(model_path, artifact_path="final_models")
                # Tag the RUN, not the artifact
                mlflow.set_tag("final_model_seed", str(seed))
                
                shutil.rmtree(model_dir)

                # Get feature importance as dictionary
                model_feature_importance = model.feature_importance(importance_type='gain')
                feature_names = self.df.drop([self.dataset["target"]]).columns

                importance_dict = {
                    feature: float(importance)
                    for feature, importance in zip(feature_names, model_feature_importance)
                }

                sorted_importance_dict = dict(sorted(importance_dict.items(), 
                    key=lambda x: x[1], 
                    reverse=True)
                )

                # Log feature importance
                mlflow.log_dict(sorted_importance_dict, f"feature_importance_{i+1}.json")

                # Aggregate feature importances
                for feature, importance in importance_dict.items():
                    if feature not in self.aggregated_feature_importance:
                        self.aggregated_feature_importance[feature] = 0.0
                    self.aggregated_feature_importance[feature] += importance

            logger.info("✅ Final models trained and logged successfully.")

            # Calculate average feature importance across all models
            avg_feature_importance = {
                feature: importance / num_seeds 
                for feature, importance in self.aggregated_feature_importance.items()
            }

            # Sort by importance (highest first)
            sorted_importance = dict(sorted(avg_feature_importance.items(), 
                                        key=lambda x: x[1], 
                                        reverse=True))

            # Log aggregated feature importance
            mlflow.log_dict(sorted_importance, "aggregated_feature_importance.json")
            logger.info(f"✅ Logged aggregated feature importance for {len(sorted_importance)} features")

            logger.info("Making predictions on test set...")
            predictions = np.mean([model.predict(self.X_test) for model in self.final_models], axis=0)

            # Calculate total forecast error
            tfe = pl.DataFrame({
                'y_true': self.y_test,
                'y_pred': predictions,
                'quantity_tn_standardized': self.global_quantity_tn_standardized_test,
                'mean': self.global_mean_values_test,
                'std': self.global_std_values_test
            }).with_columns([
                # Reverse transform predictions
                (((pl.col('quantity_tn_standardized') + pl.col('y_pred')) * pl.col('std')) + pl.col('mean')).alias('quantity_tn_pred'),
                # Reverse transform true values
                (((pl.col('quantity_tn_standardized') + pl.col('y_true')) * pl.col('std')) + pl.col('mean')).alias('quantity_tn_true')
            ]).with_columns([
                # Zero out negative predictions
                pl.col('quantity_tn_pred').clip(0.0, None).alias('quantity_tn_pred')
            ]).with_columns([
                # Calculate absolute differences
                (pl.col('quantity_tn_true') - pl.col('quantity_tn_pred')).abs().alias('abs_diff')
            ]).select([
                pl.col('abs_diff').sum().alias('numerator'),
                pl.col('quantity_tn_true').sum().alias('denominator')
            ]).with_columns([
                (pl.col('numerator') / (pl.col('denominator'))).alias('tfe')
            ])['tfe'].item()

            logger.info(f"Total Forecast Error on test set: {tfe:.4f}")

            mlflow.log_metric("test_total_forecast_error", tfe)

            mlflow.set_tag("final_training_completed", "true")

            logger.info("✅ Final training completed and logged")

    def predict_for_kaggle(self):
        """
        Make predictions for Kaggle submission.
        Use the final trained models to predict on the Kaggle dataset for the cherry compliant rows.
        Use the 12 month average for the non cherry compliant rows or the rows with problematic standardization.
        """

        with mlflow.start_run(run_name="kaggle_predictions", nested=True):
            logger.info("Retrieving product IDs for Kaggle submission...")
            kaggle_product_ids_content = self.gcp_manager.download_file_as_bytes(self.dataset["products_for_kaggle_file"])
            required_product_ids_df = pl.read_csv(BytesIO(kaggle_product_ids_content), has_header=True)
            kaggle_product_ids = required_product_ids_df.select('product_id').to_numpy().flatten()

            kaggle_to_predict = self.kaggle_dataset.filter(
                pl.col('product_id').is_in(kaggle_product_ids)
            )

            # Get the model compliant rows
            kaggle_model_compliant = kaggle_to_predict.filter(
                (pl.col('cherry_flag') == 1) & (pl.col('invalid_standardization_flag') == 0)
            )

            # Log to mlflow the proportion of compliant rows out of total rows
            compliant_count = len(kaggle_model_compliant)
            total_count = len(kaggle_to_predict)
            non_compliant_count = total_count - compliant_count
            proportion_compliant_rows = compliant_count / total_count if total_count > 0 else 0.0
            proportion_non_compliant_rows = non_compliant_count / total_count if total_count > 0 else 0.0
            mlflow.log_metric("kaggle_compliant_proportion_of_rows", proportion_compliant_rows)
            logger.info(f"Proportion of compliant rows: {proportion_compliant_rows:.4f} ({compliant_count}/{total_count})")
            mlflow.log_metric("kaggle_non_compliant_proportion_of_rows", proportion_non_compliant_rows)
            logger.info(f"Proportion of non-compliant rows: {proportion_non_compliant_rows:.4f} ({non_compliant_count}/{total_count})")

            # Log to mlflow the proportion of compliant quantity_tn_rolling_mean_11m out of total quantity_tn_rolling_mean_11m
            compliant_quantity_tn_rolling_mean_11m_sum = kaggle_model_compliant.select(
                pl.col('quantity_tn_rolling_mean_11m').sum()
            ).item()
            total_quantity_tn_rolling_mean_11m_sum = kaggle_to_predict.select(
                pl.col('quantity_tn_rolling_mean_11m').sum()
            ).item()
            non_compliant_quantity_tn_rolling_mean_11m_sum = total_quantity_tn_rolling_mean_11m_sum - compliant_quantity_tn_rolling_mean_11m_sum
            proportion_compliant_quantity_tn_rolling_mean_11m = (
                compliant_quantity_tn_rolling_mean_11m_sum / total_quantity_tn_rolling_mean_11m_sum if total_quantity_tn_rolling_mean_11m_sum > 0 else 0.0
            )
            proportion_non_compliant_quantity_tn_rolling_mean_11m = (
                non_compliant_quantity_tn_rolling_mean_11m_sum / total_quantity_tn_rolling_mean_11m_sum if total_quantity_tn_rolling_mean_11m_sum > 0 else 0.0
            )
            mlflow.log_metric("kaggle_compliant_quantity_tn_rolling_mean_11m_proportion", proportion_compliant_quantity_tn_rolling_mean_11m)
            logger.info(f"Proportion of compliant quantity_tn_rolling_mean_11m: {proportion_compliant_quantity_tn_rolling_mean_11m:.4f} ({compliant_quantity_tn_rolling_mean_11m_sum}/{total_quantity_tn_rolling_mean_11m_sum})")
            mlflow.log_metric("kaggle_non_compliant_quantity_tn_rolling_mean_11m_proportion", proportion_non_compliant_quantity_tn_rolling_mean_11m)
            logger.info(f"Proportion of non-compliant quantity_tn_rolling_mean_11m: {proportion_non_compliant_quantity_tn_rolling_mean_11m:.4f} ({non_compliant_quantity_tn_rolling_mean_11m_sum}/{total_quantity_tn_rolling_mean_11m_sum})")

            # Prepare predictions for model compliant rows
            kaggle_X = kaggle_model_compliant.drop([self.dataset["target"]]).to_numpy()
            predictions = np.mean([model.predict(kaggle_X) for model in self.final_models], axis=0)

            kaggle_product_ids_actual = kaggle_model_compliant.select('product_id').to_numpy().flatten()
            kaggle_quantity_tn_standardized = kaggle_model_compliant.select('quantity_tn_standardized').to_numpy().flatten()
            kaggle_mean = kaggle_model_compliant.select('quantity_tn_cumulative_mean').to_numpy().flatten()
            kaggle_std = kaggle_model_compliant.select('quantity_tn_cumulative_std').to_numpy().flatten()

            transformed_predictions = (
                pl.DataFrame({
                    'product_id': kaggle_product_ids_actual,
                    'quantity_tn_standardized': kaggle_quantity_tn_standardized,
                    'y_pred': predictions,
                    'mean': kaggle_mean,
                    'std': kaggle_std
                })
                .with_columns([
                    (((pl.col('quantity_tn_standardized') + pl.col('y_pred')) * pl.col('std')) + pl.col('mean')).alias('quantity_tn_pred')
                ]).with_columns([
                    # Zero out negative predictions
                    pl.col('quantity_tn_pred').clip(0.0, None).alias('tn')
                ]).select(['product_id', 'tn'])
            )

            kaggle_non_model_compliant = kaggle_to_predict.filter(
                (pl.col('cherry_flag') == 0) | (pl.col('invalid_standardization_flag') == 1)
            ).select(
                pl.col('product_id'),
                pl.col('quantity_tn_rolling_mean_11m')
                .fill_null(0)  # Replace null values with 0
                .fill_nan(0)   # Replace NaN values with 0
                .alias('tn'),
            )

            # Union the datasets and sum tn by product_id
            final_predictions = (
                transformed_predictions
                .vstack(kaggle_non_model_compliant)  # Union the two datasets
                .group_by('product_id')
                .agg(pl.col('tn').sum().alias('tn'))  # Sum tn by product_id
            )
            
            final_submission_df = (
                required_product_ids_df
                .join(final_predictions, on='product_id', how='left')
                .with_columns(pl.col('tn').fill_null(0.0))
                .select(['product_id', 'tn'])
            )

            # Save to a temp file
            submission_path = os.path.join(tempfile.gettempdir(), "kaggle_submission.csv")
            final_submission_df.write_csv(submission_path)
            
            # Log submission statistics
            mlflow.log_metric("kaggle_total_products", len(final_submission_df))
            
            # Log submission file to MLflow
            mlflow.log_artifact(submission_path, artifact_path="kaggle_submission")
            logger.info(f"Kaggle submission file logged to MLflow: {submission_path}")
            
            # Batch generate all multiplier files
            logger.info("🚀 Batch generating 18 submission variants...")

            multipliers = [round(x * 0.01, 2) for x in range(90, 100)] + [round(x * 0.01, 2) for x in range(101, 111)]

            # Create all multiplied columns at once using Polars vectorization
            multiplied_df = final_submission_df.with_columns([
                (pl.col('tn') * multiplier).alias(f'tn_x{multiplier:.2f}'.replace('.', '_'))
                for multiplier in multipliers
            ])

            # Write and log each variant
            temp_files_to_cleanup = []

            for multiplier in multipliers:
                try:
                    multiplier_str = f"{multiplier:.2f}".replace('.', '_')
                    col_name = f'tn_x{multiplier_str}'
                    
                    # Select specific multiplied column
                    variant_df = multiplied_df.select([
                        'product_id',
                        pl.col(col_name).alias('tn')
                    ])
                    
                    # Write and log
                    multiplier_filename = f"kaggle_submission_x{multiplier_str}.csv"
                    multiplier_path = os.path.join(tempfile.gettempdir(), multiplier_filename)
                    variant_df.write_csv(multiplier_path)
                    
                    artifact_path = f"kaggle_submission_multipliers/x{multiplier_str}"
                    mlflow.log_artifact(multiplier_path, artifact_path=artifact_path)
                    
                    temp_files_to_cleanup.append(multiplier_path)
                    logger.info(f"✅ Logged variant {multiplier}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed variant {multiplier}: {str(e)}")

            # Cleanup all temp files
            for temp_file in temp_files_to_cleanup + [submission_path]:
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

            # Summary metrics
            mlflow.log_metric("total_submission_variants", len(multipliers) + 1)

            logger.info(f"✅ All {len(multipliers) + 1} submission files logged to MLflow successfully!")