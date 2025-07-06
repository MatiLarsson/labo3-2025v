# dataset_generator.py
import mlflow
import duckdb
from loguru import logger
import os
import shutil
import psutil
import math
from pathlib import Path

from multinational.gcs_manager import GCSManager
from multinational.config import ProjectConfig
from multinational.utils import get_mlflow_tracking_uri, verify_mlflow_gcs_access, create_mlflow_experiment_if_not_exists

class DatasetGenerator:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.experiment_name = self.config.experiment_name
        self.dataset = self.config.dataset
        self.gcp_manager = GCSManager(self.config)
        self.sql_functions = {
            'mean': 'AVG',
            'std': 'STDDEV',
            'min': 'MIN',
            'max': 'MAX',
            'sum': 'SUM',
            'count': 'COUNT',
            'variance': 'VAR_SAMP',
            'median': 'MEDIAN',
            'skewness': 'SKEWNESS',
            'kurtosis': 'KURTOSIS'
        }
        self.minimum_windows = {
            'kurtosis': 6,
            'skewness': 6,
            'variance': 3
        }
        self.conn = duckdb.connect()
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

    def download_data(self):
        """Download necessary data files from GCS (or local read) and register them in DuckDB."""
        
        files = {
            'sell_in': 'sell-in.txt',
            'products': 'tb_productos.txt', 
            'stocks': 'tb_stocks.txt',
            'to_predict': 'product_id_to_predict_201912.txt'
        }
        
        for table_name, filename in files.items():
            # Check if script is running locally
            if self.gcp_manager.is_running_on_gcp():
                content = self.gcp_manager.download_file_as_text(filename)
                
                # Create temporary file for DuckDB
                temp_path = f"/tmp/{filename}"
                with open(temp_path, 'w') as f:
                    f.write(content)
                
                # Force immediate data loading by converting to dataframe first
                df = self.conn.query(f"SELECT DISTINCT * FROM read_csv_auto('{temp_path}')").df()
                
                # Clean up temporary file
                os.remove(temp_path)
                
                # Register the dataframe with DuckDB
                self.conn.register(table_name, df)

                logger.info(f"Downloaded and registered table '{table_name}'")
            else:
                local_dir = os.getenv('LOCAL_PROJECT_DIR', default='.')
                path = os.path.join(local_dir, 'data', filename)
                # Read local file directly
                df = self.conn.query(f"SELECT DISTINCT * FROM read_csv_auto('{path}')")
                self.conn.register(table_name, df)

                logger.info(f"Read and registered table '{table_name}'")

    def _get_testing_limiter(self):
        """
        Get SQL limiter for testing mode.
        
        Args:
            testing_mode (bool): If True, limits the dataset for faster development
            
        Returns:
            str: SQL LIMIT clause if in testing mode, empty string otherwise
        """
        if self.dataset["testing_mode"]:
            logger.info("🧪 Applying testing limits for faster development")

            testing_limiter = f"""
                AND dc.customer_id IN (SELECT customer_id FROM sell_in LIMIT {self.dataset["max_customers"]})  -- Top {self.dataset["max_customers"]} customers
                AND dp.product_id IN (SELECT product_id FROM sell_in GROUP BY product_id ORDER BY SUM(tn) DESC LIMIT {self.dataset["max_products"]})  -- Top {self.dataset["max_products"]} products by volume
            """
        else:
            testing_limiter = ""

        return testing_limiter
    
    @staticmethod
    def _generate_lag_features(
            column_name,
            max_lag,
            partition_columns
        ):
        """
        Generate SQL for multiple lag features.
      
        Args:
            column_name (str): Column name to create lag features for
            max_lag (int): Maximum number of lags to create (1 to max_lag)
            partition_columns (list): Columns to partition by
        
        Returns:
            str: SQL string with all lag features
        """
        partition_clause = f"PARTITION BY {', '.join(partition_columns)}"
        
        lag_clauses = []
        
        for lag in range(1, max_lag + 1):
            lag_clause = f"""
            -- Lag {lag}: Value from {lag} period(s) ago (captures temporal dependency)
            LAG({column_name}, {lag}) OVER (
                {partition_clause}
                ORDER BY periodo
            ) AS {column_name}_lag_{lag}"""
                
            lag_clauses.append(lag_clause)
            
        return ",\n".join(lag_clauses)
    
    @staticmethod
    def _generate_lag_features_on_standardized_values(
            z_mean_column,
            z_std_column,
            column_name,
            max_z_lag,
            partition_columns
        ):
        """
        Generate SQL for multiple z-lag features.
      
        Args:
            z_mean_column (str): Column name containing the cumulative mean for standardization
            z_std_column (str): Column name containing the cumulative std for standardization  
            column_name (str): Column name to create z-lag features for
            max_z_lag (int): Maximum number of lags to create (1 to max_z_lag)
            partition_columns (list): Columns to partition by
        
        Returns:
            str: SQL string with all standardized lag features
        """
        partition_clause = f"PARTITION BY {', '.join(partition_columns)}"
        
        z_lag_clauses = []
        
        for lag in range(1, max_z_lag + 1):
            lag_clause = f"""
            -- Standardized lag {lag}: Standardized value from {lag} period(s) ago (captures temporal dependency)
            CASE
                WHEN {z_std_column} > 0
                THEN (
                    LAG({column_name}, {lag}) OVER (
                        {partition_clause}
                        ORDER BY periodo
                    ) - {z_mean_column}
                ) / {z_std_column}
                ELSE 0
            END AS {column_name}_z_lag_{lag}"""
                
            z_lag_clauses.append(lag_clause)
            
        return ",\n".join(z_lag_clauses)

    @staticmethod
    def _get_statistic_description(stat):
        """
        Helper function to provide descriptions of statistical measures.
        """
        descriptions = {
            'mean': 'Average value over the window period',
            'median': 'Middle value (robust to outliers)',
            'std': 'Standard deviation (measure of variability)',
            'min': 'Minimum value in the window',
            'max': 'Maximum value in the window', 
            'sum': 'Total sum over the window period',
            'count': 'Number of non-null observations',
            'variance': 'Variance (squared standard deviation)',
            'skewness': 'Distribution asymmetry measure',
            'kurtosis': 'Distribution tail heaviness measure'
        }

        return descriptions.get(stat, 'Statistical measure')
    
    def _generate_rolling_statistics(
            self,
            column_name, 
            rolling_stats_window_sizes, 
            rolling_statistics,
            partition_columns
        ):
        """
        Generate SQL for comprehensive rolling window statistics including distribution shape measures.
        
        Args:
            column_name (str): Column name to create rolling features for
            rolling_stats_window_sizes (list): List of window sizes in months
            rolling_statistics (list): Statistics to calculate -> ['mean', 'median', 'std', 'min', 'max', 'sum', 'count', 'variance', 'skewness', 'kurtosis']
            partition_columns (list): Columns to partition by
        
        Returns:
            str: SQL string with all requested rolling statistics
        """
        
        partition_clause = f"PARTITION BY {', '.join(partition_columns)}"
        rolling_clauses = []
        
        for window in rolling_stats_window_sizes:
            for stat in rolling_statistics:
                stat_lower = stat.lower()
                
                # Enforce statistical best practices for minimum window sizes
                if stat_lower in self.minimum_windows:
                    min_window = self.minimum_windows[stat_lower]
                    if window < min_window:
                        continue
                
                sql_function = self.sql_functions[stat_lower]
            
                # Generate the rolling window clause
                rolling_clause = f"""
                -- {stat.capitalize()}: {self._get_statistic_description(stat_lower)} over {window} months
                {sql_function}({column_name}) OVER (
                    {partition_clause}
                    ORDER BY periodo
                    ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
                ) AS {column_name}_rolling_{stat}_{window}m"""
                
                rolling_clauses.append(rolling_clause)
        
        return ",\n".join(rolling_clauses)
    
    @staticmethod
    def _generate_anchored_delta_features_on_zlag_columns(
            base_column_name,
            max_z_lag,
            z_anchored_delta_lag_periods
        ):
        """
        Generate delta features showing change from previous periods against t0, using pre-calculated z-lag columns.
        
        Args:
            base_column_name (str): Base column name (e.g., 'quantity_tn')
            max_z_lag (int): Maximum z-lag that was calculated in previous CTE
            z_delta_lag_periods (list): Periods to calculate deltas against -> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            
        Returns:
            str: SQL string with delta features on z-lag columns
        """
        z_delta_clauses = []
        
        # Generate deltas for each z-lag column
        z_base_column_name = f"{base_column_name}_standardized"
             
        for z_delta_lag in z_anchored_delta_lag_periods:

            if z_delta_lag <= max_z_lag:  # Can't calculate delta if z_delta_lag > z_lag
                z_lag = z_delta_lag
                z_lag_column = f"{base_column_name}_z_lag_{z_lag}"
                z_delta_clause = f"""
                -- Z-Lag-Delta: Change in {z_base_column_name} over {z_delta_lag} period(s)
                {z_base_column_name} - {z_lag_column} AS {z_lag_column}_anchored_delta_z_lag_{z_delta_lag}"""
                
                z_delta_clauses.append(z_delta_clause)
        
        return ",\n".join(z_delta_clauses)
    
    @staticmethod
    def _generate_adjacent_delta_features_on_zlag_columns(
            base_column_name,
            max_z_lag,
            z_adjacent_delta_lag_periods
        ):
        """
        Generate lag delta features measuring period-over-period change in z-lagged values
        
        Args:
            base_column_name (str): Base column name (e.g., 'quantity_tn')
            max_z_lag (int): Maximum z-lag that was calculated in previous CTE
            z_adjacent_delta_lag_periods (list): Periods to calculate adjacent deltas against -> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            
        Returns:
            str: SQL string with adjacent delta features on z-lag columns
        """
        z_adjacent_delta_clauses = []
             
        for z_adjacent_delta_lag in z_adjacent_delta_lag_periods:

            if z_adjacent_delta_lag < max_z_lag:  # Can't calculate adjacent delta if z_adjacent_delta_lag >= z_lag
                z_lag = z_adjacent_delta_lag
                z_adjacent_lag_column = f"{base_column_name}_z_lag_{z_lag}"
                prev_z_adjacent_lag_column = f"{base_column_name}_z_lag_{z_lag+1}"
                z_adjacent_delta_clause = f"""
                -- Z-Lag-Adjacent-Delta: Difference between {z_adjacent_lag_column} and {prev_z_adjacent_lag_column}
                {z_adjacent_lag_column} - {prev_z_adjacent_lag_column} AS {z_adjacent_lag_column}_adjacent_delta_z_lag_{z_lag}"""
                
                z_adjacent_delta_clauses.append(z_adjacent_delta_clause)
        
        return ",\n".join(z_adjacent_delta_clauses)
    
    @staticmethod
    def _generate_anchored_ratio_features(
            column_name,
            ratio_lag_periods,
            partition_columns
        ):
        """
        Generate ratio features showing proportional change between current and previous periods.
        
        Args:
            column_name (str): Column name to calculate ratios for
            ratio_lag_periods (list): Periods to calculate ratios against -> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            partition_columns (list): Columns to partition by
            
        Returns:
            str: SQL string with ratio features
        """  
        partition_clause = f"PARTITION BY {', '.join(partition_columns)}"
        ratio_clauses = []
        
        for lag in ratio_lag_periods:
            ratio_clause = f"""
            -- Values ratio {lag}: Current value / value from {lag} period(s) ago (growth rate)
            {column_name} / NULLIF(LAG({column_name}, {lag}) OVER (
                {partition_clause}
                ORDER BY periodo
            ), 0) AS ratio_{column_name}_anchored_ratio_{lag}"""

            ratio_clauses.append(ratio_clause)
        
        return ",\n".join(ratio_clauses)
    
    @staticmethod
    def _generate_anchored_ratio_features_on_zlag_columns(
            base_column_name,
            max_z_lag,
            z_anchored_ratio_lag_periods
        ):
        """
        Generate ratio features between current and past values using pre-calculated z-lag columns.
        
        Args:
            base_column_name (str): Base column name (e.g., 'quantity_tn')
            max_z_lag (int): Maximum z-lag that was calculated in previous CTE
            z_anchored_ratio_lag_periods (list): Periods to calculate ratios against
            
        Returns:
            str: SQL string with ratio features on z-lag columns
        """
        z_anchored_ratio_clauses = []
        
        z_base_column_name = f"{base_column_name}_standardized"
            
        for z_anchored_ratio_lag in z_anchored_ratio_lag_periods:
            if z_anchored_ratio_lag <= max_z_lag:  # Can't calculate ratio if z_anchored_ratio_lag > z_lag
                z_lag = z_anchored_ratio_lag
                z_anchored_lag_column = f"{base_column_name}_z_lag_{z_lag}"
                z_anchored_ratio_clause = f"""
                -- Z-Lag-Anchored-Ratio: Ratio of {z_base_column_name} to {z_anchored_ratio_lag} period(s) ago
                {z_base_column_name} / NULLIF({z_anchored_lag_column}, 0) AS {z_anchored_lag_column}_anchored_ratio_z_lag_{z_anchored_ratio_lag}"""
                
                z_anchored_ratio_clauses.append(z_anchored_ratio_clause)
        
        return ",\n".join(z_anchored_ratio_clauses)
    
    @staticmethod
    def _generate_adjacent_ratio_features_on_zlag_columns(
            base_column_name,
            max_z_lag,
            z_adjacent_ratio_lag_periods
        ):
        """
        Generate ratio features between adjacent periods using pre-calculated z-lag columns.
        
        Args:
            base_column_name (str): Base column name (e.g., 'quantity_tn')
            max_z_lag (int): Maximum z-lag that was calculated in previous CTE
            z_adjacent_ratio_lag_periods (list): Periods to calculate ratios against
            
        Returns:
            str: SQL string with ratio features on z-lag columns
        """
        z_adjacent_ratio_clauses = []
            
        for z_adjacent_ratio_lag in z_adjacent_ratio_lag_periods:
            if z_adjacent_ratio_lag < max_z_lag:  # Can't calculate ratio if z_adjacent_ratio_lag >= z_lag
                z_lag = z_adjacent_ratio_lag
                z_adjacent_lag_column = f"{base_column_name}_z_lag_{z_lag}"
                prev_z_adjacent_lag_column = f"{base_column_name}_z_lag_{z_lag+1}"
                z_adjacent_ratio_clause = f"""
                -- Z-Lag-Adjacent-Ratio: Ratio of {z_adjacent_lag_column} to immediate previous period
                {z_adjacent_lag_column} / NULLIF({prev_z_adjacent_lag_column}, 0) AS {z_adjacent_lag_column}_adjacent_ratio_z_lag_{z_adjacent_ratio_lag}"""
                
                z_adjacent_ratio_clauses.append(z_adjacent_ratio_clause)
        
        return ",\n".join(z_adjacent_ratio_clauses)
    
    @staticmethod
    def _generate_regression_slopes(
            column_name,
            regression_slopes_window_sizes,
            partition_columns
        ):
        """
        Generate regression slope features to capture trends over rolling windows.
        
        Args:
            column_name (str): Column name to calculate slopes for
            window_sizes (list): Window sizes for slope calculation (minimum 2)
            partition_columns (list): Columns to partition by
            
        Returns:
            str: SQL string with regression slope features
        """
        partition_clause = f"PARTITION BY {', '.join(partition_columns)}"
        slope_clauses = []
        
        for window in regression_slopes_window_sizes:
            if window < 2:  # Need at least 2 points for slope
                continue
                
            slope_clause = f"""
            -- Slope {window}m: Linear regression slope over {window} months (trend direction)
            REGR_SLOPE({column_name}, periodo) OVER (
                {partition_clause}
                ORDER BY periodo
                ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
            ) AS {column_name}_slope_{window}m"""
            
            slope_clauses.append(slope_clause)
        
        return ",\n".join(slope_clauses)
    
    @staticmethod
    def _generate_regression_slopes_on_zlag_columns(
            base_column_name,
            max_z_lag,
            z_regression_slopes_window_sizes
        ):
        """
        Generate regression slope features using pre-calculated z-lag columns.
        
        Args:
            base_column_name (str): Base column name (e.g., 'quantity_tn')
            max_z_lag (int): Maximum z-lag that was calculated in previous CTE
            z_regression_slopes_window_sizes (list): Window sizes for slope calculation
            
        Returns:
            str: SQL string with regression slope features on z-lag columns
        """
        z_slope_clauses = []
            
        for window in z_regression_slopes_window_sizes:
            if window < 2 or window > max_z_lag + 1:  # Need at least 2 points, max is current + max_z_lag
                continue
                
            # Create individual value/time pairs for the regression
            value_time_pairs = []
            
            # Current standardized value at time 0
            value_time_pairs.append(f"({base_column_name}_standardized, 0)")
            
            # Historical z-lag values at negative time points
            for lag in range(1, window):
                value_time_pairs.append(f"({base_column_name}_z_lag_{lag}, -{lag})")
            
            z_slope_clause = f"""
            -- Z-Slope {window}m: Regression slope using current standardized + {window-1} z-lag columns
            (
                SELECT REGR_SLOPE(z_val, time_point)
                FROM (
                    VALUES {', '.join(value_time_pairs)}
                ) AS t(z_val, time_point)
                WHERE z_val IS NOT NULL
            ) AS {base_column_name}_z_slope_{window}m"""
            
            z_slope_clauses.append(z_slope_clause)
        
        return ",\n".join(z_slope_clauses)
    
    @staticmethod
    def _configure_duckdb_auto(conn, custom_thread_multiplier=None):
        """
        Configura DuckDB automáticamente usando el 80% de los recursos disponibles
        
        Args:
            custom_thread_multiplier: Si se especifica, usa este multiplicador para threads.
                                    Si es None, deja que DuckDB decida automáticamente.
        """
        # 1. Detectar RAM disponible (80% de la total)
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        available_ram_gb = math.floor(total_ram_gb * 0.8)
        
        # 2. Detectar espacio en disco disponible (80% del espacio libre)
        temp_dir = '/tmp/duckdb_temp'
        Path(temp_dir).mkdir(exist_ok=True)
        
        disk_usage = shutil.disk_usage(temp_dir)
        free_disk_gb = disk_usage.free / (1024**3)
        available_disk_gb = math.floor(free_disk_gb * 0.8)
        
        # 3. Detectar CPUs
        total_cpus = os.cpu_count()
        
        # 4. Configurar threads (opcional)
        if custom_thread_multiplier is not None:
            threads_to_use = round(total_cpus * custom_thread_multiplier)
            thread_config_msg = f"usando {threads_to_use} threads (manual: {custom_thread_multiplier}x)"
        else:
            threads_to_use = None
            thread_config_msg = f"auto-detectado por DuckDB (~{total_cpus} threads)"
        
        # 5. Logging de la configuración detectada
        print(f"🔧 Auto-configurando DuckDB:")
        print(f"   💾 RAM total: {total_ram_gb:.1f} GB -> usando {available_ram_gb} GB")
        print(f"   💿 Disco libre: {free_disk_gb:.1f} GB -> usando {available_disk_gb} GB")
        print(f"   🔄 CPUs: {total_cpus} -> {thread_config_msg}")
        print(f"   📁 Directorio temporal: {temp_dir}")
        
        # 6. Aplicar configuración
        conn.execute(f"PRAGMA memory_limit='{available_ram_gb}GiB'")
        conn.execute(f"PRAGMA max_temp_directory_size='{available_disk_gb}GiB'")
        conn.execute(f"PRAGMA temp_directory='{temp_dir}'")
        
        # Solo configurar threads si se especifica manualmente
        if threads_to_use is not None:
            conn.execute(f"PRAGMA threads={threads_to_use}")
        
        # 7. Configuraciones adicionales para optimización
        conn.execute("PRAGMA enable_progress_bar=true")
        conn.execute("PRAGMA preserve_insertion_order=false")
        
        print("✅ DuckDB configurado automáticamente")
        
        return {
            'memory_limit_gb': available_ram_gb,
            'temp_directory_size_gb': available_disk_gb,
            'threads': threads_to_use if threads_to_use else f"auto (~{total_cpus})",
            'temp_directory': temp_dir
        }

    def generate_dataset(self):
        """
        Generate the complete SQL query for the dataset.
        
        Args:
            testing_mode (bool): If True, adds limitations for faster development
            
        Returns:
            str: Complete SQL query for dataset creation
        """

        duck_db_auto_config = self._configure_duckdb_auto(self.conn)
        logger.info(f"🔧 DuckDB auto-configuration: {duck_db_auto_config}")

        with mlflow.start_run(run_name="dataset"):
            # Check if a dataset already exists for this experiment in MLflow, if so, skip generation
            existing_runs = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                filter_string="tags.dataset_generated = 'true'",
                max_results=1
            )

            if not existing_runs.empty:
                logger.info("📂 Dataset already generated for this experiment. Skipping generation.")
                return

            # Log the entire config
            config_dict = self.config.model_dump()
            # MLflow has limits on parameter values, so we need to handle large configs
            for key, value in config_dict.items():
                try:
                    if isinstance(value, (dict, list)):
                        mlflow.log_param(key, str(value)[:250])  # Truncate if too long
                    else:
                        mlflow.log_param(key, str(value))
                except Exception as e:
                    logger.warning(f"Could not log parameter {key}: {e}")

            testing_limiter = self._get_testing_limiter()
            
            logger.info("🛠️ Generating dataset")

            self.conn.execute(f"""
            CREATE OR REPLACE TABLE dataset AS (
                WITH date_bounds AS (
                    -- Calculate the full date range
                    SELECT 
                        MIN(periodo) as min_periodo,
                        MAX(periodo) as max_periodo
                    FROM sell_in
                ),
                period_series AS (
                    -- Generate all possible periods in our dataset
                    SELECT CAST(strftime(periodo_date, '%Y%m') AS BIGINT) AS periodo
                    FROM (
                        SELECT unnest(generate_series(
                            strptime(CAST((SELECT min_periodo FROM date_bounds) AS VARCHAR), '%Y%m'),
                            strptime(CAST((SELECT max_periodo FROM date_bounds) AS VARCHAR), '%Y%m'),
                            INTERVAL '1 month'
                        )) AS periodo_date
                    )
                ),
                customer_activity AS (
                    -- Track when each customer became active in our system for the first and last time
                    SELECT
                        customer_id,
                        MIN(periodo) AS customer_first_active_period,
                        MAX(periodo) AS customer_last_active_period
                    FROM sell_in
                    GROUP BY customer_id
                ),
                product_activity AS (
                    -- Track when each product became active in our system for the first and last time
                    SELECT
                        product_id,
                        MIN(periodo) AS product_first_active_period,
                        MAX(periodo) AS product_last_active_period
                    FROM sell_in
                    GROUP BY product_id
                ),
                cat1_aggregates AS (
                    SELECT 
                        p.cat1,
                        si.periodo,
                        SUM(COALESCE(si.tn, 0)) AS cat1_total_tn,
                        SUM(COALESCE(si.cust_request_qty, 0)) AS cat1_total_cust_request_qty,
                        SUM(COALESCE(si.cust_request_tn, 0)) AS cat1_total_cust_request_tn,
                        SUM(COALESCE(s.stock_final, 0)) AS cat1_total_stock_final
                    FROM sell_in si
                    LEFT JOIN products p ON si.product_id = p.product_id
                    LEFT JOIN stocks s ON si.product_id = s.product_id AND si.periodo = s.periodo
                    WHERE p.cat1 IS NOT NULL
                    GROUP BY p.cat1, si.periodo
                ),
                cat2_aggregates AS (
                    SELECT 
                        p.cat2,
                        si.periodo,
                        SUM(COALESCE(si.tn, 0)) AS cat2_total_tn,
                        SUM(COALESCE(si.cust_request_qty, 0)) AS cat2_total_cust_request_qty,
                        SUM(COALESCE(si.cust_request_tn, 0)) AS cat2_total_cust_request_tn,
                        SUM(COALESCE(s.stock_final, 0)) AS cat2_total_stock_final
                    FROM sell_in si
                    LEFT JOIN products p ON si.product_id = p.product_id
                    LEFT JOIN stocks s ON si.product_id = s.product_id AND si.periodo = s.periodo
                    WHERE p.cat2 IS NOT NULL
                    GROUP BY p.cat2, si.periodo
                ),
                cat3_aggregates AS (
                    SELECT 
                        p.cat3,
                        si.periodo,
                        SUM(COALESCE(si.tn, 0)) AS cat3_total_tn,
                        SUM(COALESCE(si.cust_request_qty, 0)) AS cat3_total_cust_request_qty,
                        SUM(COALESCE(si.cust_request_tn, 0)) AS cat3_total_cust_request_tn,
                        SUM(COALESCE(s.stock_final, 0)) AS cat3_total_stock_final
                    FROM sell_in si
                    LEFT JOIN products p ON si.product_id = p.product_id
                    LEFT JOIN stocks s ON si.product_id = s.product_id AND si.periodo = s.periodo
                    WHERE p.cat3 IS NOT NULL
                    GROUP BY p.cat3, si.periodo
                ),
                brand_aggregates AS (
                    SELECT 
                        p.brand,
                        si.periodo,
                        SUM(COALESCE(si.tn, 0)) AS brand_total_tn,
                        SUM(COALESCE(si.cust_request_qty, 0)) AS brand_total_cust_request_qty,
                        SUM(COALESCE(si.cust_request_tn, 0)) AS brand_total_cust_request_tn,
                        SUM(COALESCE(s.stock_final, 0)) AS brand_total_stock_final
                    FROM sell_in si
                    LEFT JOIN products p ON si.product_id = p.product_id
                    LEFT JOIN stocks s ON si.product_id = s.product_id AND si.periodo = s.periodo
                    WHERE p.brand IS NOT NULL
                    GROUP BY p.brand, si.periodo
                ),
                descripcion_aggregates AS (
                    SELECT 
                        p.descripcion,
                        si.periodo,
                        SUM(COALESCE(si.tn, 0)) AS descripcion_total_tn,
                        SUM(COALESCE(si.cust_request_qty, 0)) AS descripcion_total_cust_request_qty,
                        SUM(COALESCE(si.cust_request_tn, 0)) AS descripcion_total_cust_request_tn,
                        SUM(COALESCE(s.stock_final, 0)) AS descripcion_total_stock_final
                    FROM sell_in si
                    LEFT JOIN products p ON si.product_id = p.product_id
                    LEFT JOIN stocks s ON si.product_id = s.product_id AND si.periodo = s.periodo
                    WHERE p.descripcion IS NOT NULL
                    GROUP BY p.descripcion, si.periodo
                ),
                product_aggregates AS (
                    SELECT 
                        si.product_id,
                        si.periodo,
                        SUM(COALESCE(si.tn, 0)) AS product_total_tn,
                        SUM(COALESCE(si.cust_request_qty, 0)) AS product_total_cust_request_qty,
                        SUM(COALESCE(si.cust_request_tn, 0)) AS product_total_cust_request_tn,
                        SUM(COALESCE(s.stock_final, 0)) AS product_total_stock_final
                    FROM sell_in si
                    LEFT JOIN stocks s ON si.product_id = s.product_id AND si.periodo = s.periodo
                    GROUP BY si.product_id, si.periodo
                ),
                customer_aggregates AS (
                    SELECT 
                        si.customer_id,
                        si.periodo,
                        SUM(COALESCE(si.tn, 0)) AS customer_total_tn,
                        SUM(COALESCE(si.cust_request_qty, 0)) AS customer_total_cust_request_qty,
                        SUM(COALESCE(si.cust_request_tn, 0)) AS customer_total_cust_request_tn,
                        SUM(COALESCE(s.stock_final, 0)) AS customer_total_stock_final
                    FROM sell_in si
                    LEFT JOIN stocks s ON si.product_id = s.product_id AND si.periodo = s.periodo
                    GROUP BY si.customer_id, si.periodo
                ),
                active_product_customer_period AS (
                    SELECT 
                        dp.product_id,
                        dc.customer_id,
                        ps.periodo,
                        
                        p.cat1, p.cat2, p.cat3, p.brand, p.sku_size, p.descripcion,
                        
                        -- Dummy feature for plan precios cuidados
                        MAX(COALESCE(si.plan_precios_cuidados, 0)) AS dummy_plan_precios_cuidados,
                        
                        -- Aggregate quantities for the product and customer in this period
                        SUM(COALESCE(si.cust_request_qty, 0)) AS quantity_cust_request_qty,
                        SUM(COALESCE(si.cust_request_tn, 0)) AS quantity_cust_request_tn,
                        SUM(COALESCE(si.tn, 0)) AS quantity_tn,
                        SUM(s.stock_final) AS quantity_stock_final
                        
                    FROM (SELECT DISTINCT product_id FROM products) dp
                    CROSS JOIN (SELECT DISTINCT customer_id FROM sell_in) dc
                    CROSS JOIN period_series ps
                    LEFT JOIN product_activity pa ON dp.product_id = pa.product_id
                    LEFT JOIN customer_activity ca ON dc.customer_id = ca.customer_id
                    LEFT JOIN sell_in si ON dp.product_id = si.product_id 
                        AND dc.customer_id = si.customer_id
                        AND ps.periodo = si.periodo
                    LEFT JOIN products p ON dp.product_id = p.product_id
                    LEFT JOIN stocks s ON dp.product_id = s.product_id
                        AND ps.periodo = s.periodo
                    -- Only include periods where the customer and product overlap with their active periods
                    WHERE ps.periodo >= GREATEST(ca.customer_first_active_period, pa.product_first_active_period)
                        AND ps.periodo <= CAST(strftime('%Y%m', date_add(
                            strptime(CAST(LEAST(ca.customer_last_active_period, pa.product_last_active_period) AS VARCHAR), '%Y%m'),
                            INTERVAL '{self.dataset["future_periods_extension"]} months'
                        )) AS BIGINT)
                        {testing_limiter}
                    -- Only include pairs of product_id and customer_id that had at least one sell_in record
                        AND EXISTS (
                            SELECT 1
                            FROM sell_in si2
                            WHERE si2.product_id = dp.product_id
                                AND si2.customer_id = dc.customer_id
                        )
                    GROUP BY
                        dp.product_id, dc.customer_id, ps.periodo,
                        p.cat1, p.cat2, p.cat3, p.brand, p.sku_size, p.descripcion
                ),
                product_customer_period_with_aggregates AS (
                    SELECT 
                        apcp.*,
                        
                        -- Aggregate features with cumulative/lag capabilities
                        -- Cat1 aggregates
                        COALESCE(c1a.cat1_total_tn, 0) AS cat1_total_tn,
                        COALESCE(c1a.cat1_total_cust_request_qty, 0) AS cat1_total_cust_request_qty,
                        COALESCE(c1a.cat1_total_cust_request_tn, 0) AS cat1_total_cust_request_tn,
                        COALESCE(c1a.cat1_total_stock_final, 0) AS cat1_total_stock_final,
                        
                        -- Cat2 aggregates
                        COALESCE(c2a.cat2_total_tn, 0) AS cat2_total_tn,
                        COALESCE(c2a.cat2_total_cust_request_qty, 0) AS cat2_total_cust_request_qty,
                        COALESCE(c2a.cat2_total_cust_request_tn, 0) AS cat2_total_cust_request_tn,
                        COALESCE(c2a.cat2_total_stock_final, 0) AS cat2_total_stock_final,
                        
                        -- Cat3 aggregates
                        COALESCE(c3a.cat3_total_tn, 0) AS cat3_total_tn,
                        COALESCE(c3a.cat3_total_cust_request_qty, 0) AS cat3_total_cust_request_qty,
                        COALESCE(c3a.cat3_total_cust_request_tn, 0) AS cat3_total_cust_request_tn,
                        COALESCE(c3a.cat3_total_stock_final, 0) AS cat3_total_stock_final,
                        
                        -- Brand aggregates
                        COALESCE(ba.brand_total_tn, 0) AS brand_total_tn,
                        COALESCE(ba.brand_total_cust_request_qty, 0) AS brand_total_cust_request_qty,
                        COALESCE(ba.brand_total_cust_request_tn, 0) AS brand_total_cust_request_tn,
                        COALESCE(ba.brand_total_stock_final, 0) AS brand_total_stock_final,
                        
                        -- Descripcion aggregates
                        COALESCE(da.descripcion_total_tn, 0) AS descripcion_total_tn,
                        COALESCE(da.descripcion_total_cust_request_qty, 0) AS descripcion_total_cust_request_qty,
                        COALESCE(da.descripcion_total_cust_request_tn, 0) AS descripcion_total_cust_request_tn,
                        COALESCE(da.descripcion_total_stock_final, 0) AS descripcion_total_stock_final,
                        
                        -- Product aggregates
                        COALESCE(pa.product_total_tn, 0) AS product_total_tn,
                        COALESCE(pa.product_total_cust_request_qty, 0) AS product_total_cust_request_qty,
                        COALESCE(pa.product_total_cust_request_tn, 0) AS product_total_cust_request_tn,
                        COALESCE(pa.product_total_stock_final, 0) AS product_total_stock_final,
                        
                        -- Customer aggregates
                        COALESCE(ca.customer_total_tn, 0) AS customer_total_tn,
                        COALESCE(ca.customer_total_cust_request_qty, 0) AS customer_total_cust_request_qty,
                        COALESCE(ca.customer_total_cust_request_tn, 0) AS customer_total_cust_request_tn,
                        COALESCE(ca.customer_total_stock_final, 0) AS customer_total_stock_final,
                        
                        -- Proportional features (individual quantities as proportion of group totals)

                        -- quantity_tn proportions
                        CASE 
                            WHEN c1a.cat1_total_tn > 0 
                            THEN apcp.quantity_tn / c1a.cat1_total_tn 
                            ELSE NULL
                        END AS quantity_tn_prop_of_cat1,
                        CASE 
                            WHEN c2a.cat2_total_tn > 0 
                            THEN apcp.quantity_tn / c2a.cat2_total_tn 
                            ELSE NULL
                        END AS quantity_tn_prop_of_cat2,
                        CASE 
                            WHEN c3a.cat3_total_tn > 0 
                            THEN apcp.quantity_tn / c3a.cat3_total_tn 
                            ELSE NULL
                        END AS quantity_tn_prop_of_cat3,
                        CASE 
                            WHEN ba.brand_total_tn > 0 
                            THEN apcp.quantity_tn / ba.brand_total_tn 
                            ELSE NULL
                        END AS quantity_tn_prop_of_brand,
                        CASE 
                            WHEN da.descripcion_total_tn > 0 
                            THEN apcp.quantity_tn / da.descripcion_total_tn 
                            ELSE NULL
                        END AS quantity_tn_prop_of_descripcion,
                        CASE 
                            WHEN pa.product_total_tn > 0 
                            THEN apcp.quantity_tn / pa.product_total_tn 
                            ELSE NULL
                        END AS quantity_tn_prop_of_product,
                        CASE 
                            WHEN ca.customer_total_tn > 0 
                            THEN apcp.quantity_tn / ca.customer_total_tn 
                            ELSE NULL
                        END AS quantity_tn_prop_of_customer,

                        -- quantity_cust_request_qty proportions

                        CASE
                            WHEN c1a.cat1_total_cust_request_qty > 0
                            THEN apcp.quantity_cust_request_qty / c1a.cat1_total_cust_request_qty
                            ELSE NULL
                        END AS quantity_cust_request_qty_prop_of_cat1,
                        CASE
                            WHEN c2a.cat2_total_cust_request_qty > 0
                            THEN apcp.quantity_cust_request_qty / c2a.cat2_total_cust_request_qty
                            ELSE NULL
                        END AS quantity_cust_request_qty_prop_of_cat2,
                        CASE
                            WHEN c3a.cat3_total_cust_request_qty > 0
                            THEN apcp.quantity_cust_request_qty / c3a.cat3_total_cust_request_qty
                            ELSE NULL
                        END AS quantity_cust_request_qty_prop_of_cat3,
                        CASE
                            WHEN ba.brand_total_cust_request_qty > 0
                            THEN apcp.quantity_cust_request_qty / ba.brand_total_cust_request_qty
                            ELSE NULL
                        END AS quantity_cust_request_qty_prop_of_brand,
                        CASE
                            WHEN da.descripcion_total_cust_request_qty > 0
                            THEN apcp.quantity_cust_request_qty / da.descripcion_total_cust_request_qty
                            ELSE NULL
                        END AS quantity_cust_request_qty_prop_of_descripcion,
                        CASE
                            WHEN pa.product_total_cust_request_qty > 0
                            THEN apcp.quantity_cust_request_qty / pa.product_total_cust_request_qty
                            ELSE NULL
                        END AS quantity_cust_request_qty_prop_of_product,
                        CASE
                            WHEN ca.customer_total_cust_request_qty > 0
                            THEN apcp.quantity_cust_request_qty / ca.customer_total_cust_request_qty
                            ELSE NULL
                        END AS quantity_cust_request_qty_prop_of_customer,
                        
                        -- quantity_cust_request_tn proportions
                        CASE
                            WHEN c1a.cat1_total_cust_request_tn > 0
                            THEN apcp.quantity_cust_request_tn / c1a.cat1_total_cust_request_tn
                            ELSE NULL
                        END AS quantity_cust_request_tn_prop_of_cat1,
                        CASE
                            WHEN c2a.cat2_total_cust_request_tn > 0
                            THEN apcp.quantity_cust_request_tn / c2a.cat2_total_cust_request_tn
                            ELSE NULL
                        END AS quantity_cust_request_tn_prop_of_cat2,
                        CASE
                            WHEN c3a.cat3_total_cust_request_tn > 0
                            THEN apcp.quantity_cust_request_tn / c3a.cat3_total_cust_request_tn
                            ELSE NULL
                        END AS quantity_cust_request_tn_prop_of_cat3,
                        CASE
                            WHEN ba.brand_total_cust_request_tn > 0
                            THEN apcp.quantity_cust_request_tn / ba.brand_total_cust_request_tn
                            ELSE NULL
                        END AS quantity_cust_request_tn_prop_of_brand,
                        CASE
                            WHEN da.descripcion_total_cust_request_tn > 0
                            THEN apcp.quantity_cust_request_tn / da.descripcion_total_cust_request_tn
                            ELSE NULL
                        END AS quantity_cust_request_tn_prop_of_descripcion,
                        CASE
                            WHEN pa.product_total_cust_request_tn > 0
                            THEN apcp.quantity_cust_request_tn / pa.product_total_cust_request_tn
                            ELSE NULL
                        END AS quantity_cust_request_tn_prop_of_product,
                        CASE
                            WHEN ca.customer_total_cust_request_tn > 0
                            THEN apcp.quantity_cust_request_tn / ca.customer_total_cust_request_tn
                            ELSE NULL
                        END AS quantity_cust_request_tn_prop_of_customer,

                        -- quantity_stock_final proportions
                        CASE
                            WHEN c1a.cat1_total_stock_final > 0
                            THEN apcp.quantity_stock_final / c1a.cat1_total_stock_final
                            ELSE NULL
                        END AS quantity_stock_final_prop_of_cat1,
                        CASE
                            WHEN c2a.cat2_total_stock_final > 0
                            THEN apcp.quantity_stock_final / c2a.cat2_total_stock_final
                            ELSE NULL
                        END AS quantity_stock_final_prop_of_cat2,
                        CASE
                            WHEN c3a.cat3_total_stock_final > 0
                            THEN apcp.quantity_stock_final / c3a.cat3_total_stock_final
                            ELSE NULL
                        END AS quantity_stock_final_prop_of_cat3,
                        CASE
                            WHEN ba.brand_total_stock_final > 0
                            THEN apcp.quantity_stock_final / ba.brand_total_stock_final
                            ELSE NULL
                        END AS quantity_stock_final_prop_of_brand,
                        CASE
                            WHEN da.descripcion_total_stock_final > 0
                            THEN apcp.quantity_stock_final / da.descripcion_total_stock_final
                            ELSE NULL
                        END AS quantity_stock_final_prop_of_descripcion,
                        CASE
                            WHEN pa.product_total_stock_final > 0
                            THEN apcp.quantity_stock_final / pa.product_total_stock_final
                            ELSE NULL
                        END AS quantity_stock_final_prop_of_product,
                        CASE
                            WHEN ca.customer_total_stock_final > 0
                            THEN apcp.quantity_stock_final / ca.customer_total_stock_final
                            ELSE NULL
                        END AS quantity_stock_final_prop_of_customer
                        
                    FROM active_product_customer_period apcp
                    LEFT JOIN cat1_aggregates c1a ON apcp.cat1 = c1a.cat1 AND apcp.periodo = c1a.periodo
                    LEFT JOIN cat2_aggregates c2a ON apcp.cat2 = c2a.cat2 AND apcp.periodo = c2a.periodo
                    LEFT JOIN cat3_aggregates c3a ON apcp.cat3 = c3a.cat3 AND apcp.periodo = c3a.periodo
                    LEFT JOIN brand_aggregates ba ON apcp.brand = ba.brand AND apcp.periodo = ba.periodo
                    LEFT JOIN descripcion_aggregates da ON apcp.descripcion = da.descripcion AND apcp.periodo = da.periodo
                    LEFT JOIN product_aggregates pa ON apcp.product_id = pa.product_id AND apcp.periodo = pa.periodo
                    LEFT JOIN customer_aggregates ca ON apcp.customer_id = ca.customer_id AND apcp.periodo = ca.periodo
                ),
                product_customer_period_extras AS (
                    SELECT
                        product_id,
                        customer_id,
                        periodo,

                        -- Dummy feature for problematic periods
                        CASE
                            WHEN (periodo = 201908 OR periodo = 201910) THEN 1
                            ELSE 0
                        END AS problematic_period_flag,
                        
                        -- Date features extracted from periodo
                        EXTRACT(MONTH FROM strptime(CAST(periodo AS VARCHAR), '%Y%m')) AS month_number,
                        EXTRACT(YEAR FROM strptime(CAST(periodo AS VARCHAR), '%Y%m')) AS year_number,
                        EXTRACT(QUARTER FROM strptime(CAST(periodo AS VARCHAR), '%Y%m')) AS quarter_number,
                        CASE 
                            WHEN EXTRACT(MONTH FROM strptime(CAST(periodo AS VARCHAR), '%Y%m')) <= 6 THEN 1 
                            ELSE 2 
                        END AS semester_number,
                        
                        -- iperiodo (from 1 to 36) according to min and max periodo available in period_series
                        ROW_NUMBER() OVER (ORDER BY periodo) AS iperiodo,
                        
                        -- Target column: quantity_tn_target (quantity_tn shifted by 2 periods forward)
                        COALESCE(LEAD(quantity_tn, 2, 0) OVER (PARTITION BY product_id, customer_id ORDER BY periodo), 0) AS quantity_tn_target
                        
                    FROM product_customer_period_with_aggregates
                ),
                product_customer_period_cumulative_standardization_stats AS (
                    SELECT
                        product_id,
                        customer_id,
                        periodo,
                                            
                        -- Cumulative stats for quantity_cust_request_qty
                        
                        -- Cumulative Mean: For standardization (z-score calculation)
                        AVG(quantity_cust_request_qty) OVER (
                            PARTITION BY product_id, customer_id
                            ORDER BY periodo
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ) AS quantity_cust_request_qty_cumulative_mean,
                        -- Cumulative Std: For standardization (z-score calculation)
                        COALESCE(
                            STDDEV(quantity_cust_request_qty) OVER (
                                PARTITION BY product_id, customer_id
                                ORDER BY periodo
                                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                            )
                            , 0
                        ) AS quantity_cust_request_qty_cumulative_std,
                                            
                        -- Cumulative stats for quantity_cust_request_tn
                        
                        -- Cumulative Mean: For standardization (z-score calculation)
                        AVG(quantity_cust_request_tn) OVER (
                            PARTITION BY product_id, customer_id
                            ORDER BY periodo
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ) AS quantity_cust_request_tn_cumulative_mean,
                        -- Cumulative Std: For standardization (z-score calculation)
                        COALESCE(
                            STDDEV(quantity_cust_request_tn) OVER (
                                PARTITION BY product_id, customer_id
                                ORDER BY periodo
                                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                            )
                            , 0
                        ) AS quantity_cust_request_tn_cumulative_std,
                                            
                        -- Cumulative stats for quantity_tn
                        
                        -- Cumulative Mean: For standardization (z-score calculation)
                        AVG(quantity_tn) OVER (
                            PARTITION BY product_id, customer_id
                            ORDER BY periodo
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ) AS quantity_tn_cumulative_mean,
                        -- Cumulative Std: For standardization (z-score calculation)
                        COALESCE(
                            STDDEV(quantity_tn) OVER (
                                PARTITION BY product_id, customer_id
                                ORDER BY periodo
                                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                            )
                            , 0
                        ) AS quantity_tn_cumulative_std,
                        
                        -- Cumulative stats for quantity_stock_final
                        AVG(quantity_stock_final) OVER (
                            PARTITION BY product_id, customer_id
                            ORDER BY periodo
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ) AS quantity_stock_final_cumulative_mean,
                        COALESCE(
                            STDDEV(quantity_stock_final) OVER (
                                PARTITION BY product_id, customer_id
                                ORDER BY periodo
                                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                            )
                            , 0
                        ) AS quantity_stock_final_cumulative_std
                                            
                    FROM product_customer_period_with_aggregates
                ),
                raw_and_standardized_product_customer_quantities AS (
                    SELECT 
                        pcpwa.product_id,
                        pcpwa.customer_id,
                        pcpwa.periodo,
                        pcpwa.cat1, pcpwa.cat2, pcpwa.cat3, pcpwa.brand, pcpwa.sku_size, pcpwa.descripcion,
                        pcpwa.dummy_plan_precios_cuidados,
                        pcpe.problematic_period_flag,
                        pcpe.month_number,
                        pcpe.year_number,
                        pcpe.quarter_number,
                        pcpe.semester_number,
                        pcpe.iperiodo,

                        -- Aggregate features
                        pcpwa.cat1_total_tn, pcpwa.cat1_total_cust_request_qty, pcpwa.cat1_total_cust_request_tn, pcpwa.cat1_total_stock_final,
                        pcpwa.cat2_total_tn, pcpwa.cat2_total_cust_request_qty, pcpwa.cat2_total_cust_request_tn, pcpwa.cat2_total_stock_final,
                        pcpwa.cat3_total_tn, pcpwa.cat3_total_cust_request_qty, pcpwa.cat3_total_cust_request_tn, pcpwa.cat3_total_stock_final,
                        pcpwa.brand_total_tn, pcpwa.brand_total_cust_request_qty, pcpwa.brand_total_cust_request_tn, pcpwa.brand_total_stock_final,
                        pcpwa.descripcion_total_tn, pcpwa.descripcion_total_cust_request_qty, pcpwa.descripcion_total_cust_request_tn, pcpwa.descripcion_total_stock_final,
                        pcpwa.product_total_tn, pcpwa.product_total_cust_request_qty, pcpwa.product_total_cust_request_tn, pcpwa.product_total_stock_final,
                        pcpwa.customer_total_tn, pcpwa.customer_total_cust_request_qty, pcpwa.customer_total_cust_request_tn, pcpwa.customer_total_stock_final,

                        -- Proportional features

                        -- quantity_tn proportions

                        pcpwa.quantity_tn_prop_of_cat1, pcpwa.quantity_tn_prop_of_cat2, pcpwa.quantity_tn_prop_of_cat3,
                        pcpwa.quantity_tn_prop_of_brand, pcpwa.quantity_tn_prop_of_descripcion, 
                        pcpwa.quantity_tn_prop_of_product, pcpwa.quantity_tn_prop_of_customer,

                        -- quantity_cust_request_qty proportions
                        pcpwa.quantity_cust_request_qty_prop_of_cat1, pcpwa.quantity_cust_request_qty_prop_of_cat2,
                        pcpwa.quantity_cust_request_qty_prop_of_cat3, pcpwa.quantity_cust_request_qty_prop_of_brand,
                        pcpwa.quantity_cust_request_qty_prop_of_descripcion, pcpwa.quantity_cust_request_qty_prop_of_product,
                        pcpwa.quantity_cust_request_qty_prop_of_customer,

                        -- quantity_cust_request_tn proportions
                        pcpwa.quantity_cust_request_tn_prop_of_cat1, pcpwa.quantity_cust_request_tn_prop_of_cat2,
                        pcpwa.quantity_cust_request_tn_prop_of_cat3, pcpwa.quantity_cust_request_tn_prop_of_brand,
                        pcpwa.quantity_cust_request_tn_prop_of_descripcion, pcpwa.quantity_cust_request_tn_prop_of_product,
                        pcpwa.quantity_cust_request_tn_prop_of_customer,

                        -- quantity_stock_final proportions
                        pcpwa.quantity_stock_final_prop_of_cat1, pcpwa.quantity_stock_final_prop_of_cat2,
                        pcpwa.quantity_stock_final_prop_of_cat3, pcpwa.quantity_stock_final_prop_of_brand,
                        pcpwa.quantity_stock_final_prop_of_descripcion, pcpwa.quantity_stock_final_prop_of_product,
                        pcpwa.quantity_stock_final_prop_of_customer,
                        
                        -- Standardization stats
                        pcpcss.quantity_cust_request_qty_cumulative_mean,
                        pcpcss.quantity_cust_request_qty_cumulative_std,
                        pcpcss.quantity_cust_request_tn_cumulative_mean,
                        pcpcss.quantity_cust_request_tn_cumulative_std,
                        pcpcss.quantity_tn_cumulative_mean,
                        pcpcss.quantity_tn_cumulative_std,
                        pcpcss.quantity_stock_final_cumulative_mean,
                        pcpcss.quantity_stock_final_cumulative_std,
                        
                        -- Raw quantities
                        pcpwa.quantity_cust_request_qty,
                        pcpwa.quantity_cust_request_tn,
                        pcpwa.quantity_tn, 
                        pcpwa.quantity_stock_final,
                        
                        -- Standardized quantities
                        CASE 
                            WHEN pcpcss.quantity_cust_request_qty_cumulative_std > 0 
                            THEN (pcpwa.quantity_cust_request_qty - pcpcss.quantity_cust_request_qty_cumulative_mean) / pcpcss.quantity_cust_request_qty_cumulative_std
                            ELSE 0 
                        END AS quantity_cust_request_qty_standardized,
                        
                        CASE 
                            WHEN pcpcss.quantity_cust_request_tn_cumulative_std > 0 
                            THEN (pcpwa.quantity_cust_request_tn - pcpcss.quantity_cust_request_tn_cumulative_mean) / pcpcss.quantity_cust_request_tn_cumulative_std
                            ELSE 0 
                        END AS quantity_cust_request_tn_standardized,
                        
                        CASE 
                            WHEN pcpcss.quantity_tn_cumulative_std > 0 
                            THEN (pcpwa.quantity_tn - pcpcss.quantity_tn_cumulative_mean) / pcpcss.quantity_tn_cumulative_std
                            ELSE 0
                        END AS quantity_tn_standardized,
                        
                        CASE 
                            WHEN pcpcss.quantity_tn_cumulative_std > 0 
                            THEN (pcpe.quantity_tn_target - pcpcss.quantity_tn_cumulative_mean) / pcpcss.quantity_tn_cumulative_std
                            ELSE 0
                        END AS quantity_tn_target_standardized,
                        
                        CASE
                            WHEN pcpcss.quantity_stock_final_cumulative_std > 0
                            THEN (pcpwa.quantity_stock_final - pcpcss.quantity_stock_final_cumulative_mean) / pcpcss.quantity_stock_final_cumulative_std
                            ELSE 0
                        END AS quantity_stock_final_standardized
                        
                    FROM product_customer_period_with_aggregates pcpwa
                    LEFT JOIN product_customer_period_cumulative_standardization_stats pcpcss ON pcpwa.product_id = pcpcss.product_id 
                        AND pcpwa.customer_id = pcpcss.customer_id
                        AND pcpwa.periodo = pcpcss.periodo
                    LEFT JOIN product_customer_period_extras pcpe ON pcpwa.product_id = pcpe.product_id
                        AND pcpwa.customer_id = pcpe.customer_id
                        AND pcpwa.periodo = pcpe.periodo
                ),
                fixed_standardized_lag_series AS (
                    SELECT
                        product_id,
                        customer_id,
                        periodo,
                        cat1, cat2, cat3, brand, sku_size, descripcion,
                                                
                        -- Dummy features
                        dummy_plan_precios_cuidados,
                        problematic_period_flag,

                        -- Date features
                        month_number, year_number, quarter_number, semester_number, iperiodo,
                            
                        -- Raw quantities
                        quantity_cust_request_qty,
                        quantity_cust_request_tn,
                        quantity_tn,
                        quantity_stock_final,
                            
                        -- Aggregate features
                        cat1_total_tn, cat1_total_cust_request_qty, cat1_total_cust_request_tn, cat1_total_stock_final,
                        cat2_total_tn, cat2_total_cust_request_qty, cat2_total_cust_request_tn, cat2_total_stock_final,
                        cat3_total_tn, cat3_total_cust_request_qty, cat3_total_cust_request_tn, cat3_total_stock_final,
                        brand_total_tn, brand_total_cust_request_qty, brand_total_cust_request_tn, brand_total_stock_final,
                        descripcion_total_tn, descripcion_total_cust_request_qty, descripcion_total_cust_request_tn, descripcion_total_stock_final,
                        product_total_tn, product_total_cust_request_qty, product_total_cust_request_tn, product_total_stock_final,
                        customer_total_tn, customer_total_cust_request_qty, customer_total_cust_request_tn, customer_total_stock_final,

                        -- Proportional features
                        quantity_tn_prop_of_cat1,
                        quantity_tn_prop_of_cat2,
                        quantity_tn_prop_of_cat3,
                        quantity_tn_prop_of_brand,
                        quantity_tn_prop_of_descripcion, 
                        quantity_tn_prop_of_product,
                        quantity_tn_prop_of_customer,
                        quantity_cust_request_qty_prop_of_cat1,
                        quantity_cust_request_qty_prop_of_cat2,
                        quantity_cust_request_qty_prop_of_cat3,
                        quantity_cust_request_qty_prop_of_brand,
                        quantity_cust_request_qty_prop_of_descripcion,
                        quantity_cust_request_qty_prop_of_product,
                        quantity_cust_request_qty_prop_of_customer,
                        quantity_cust_request_tn_prop_of_cat1,
                        quantity_cust_request_tn_prop_of_cat2,
                        quantity_cust_request_tn_prop_of_cat3, 
                        quantity_cust_request_tn_prop_of_brand,
                        quantity_cust_request_tn_prop_of_descripcion,
                        quantity_cust_request_tn_prop_of_product,
                        quantity_cust_request_tn_prop_of_customer,
                        quantity_stock_final_prop_of_cat1,
                        quantity_stock_final_prop_of_cat2,
                        quantity_stock_final_prop_of_cat3,
                        quantity_stock_final_prop_of_brand,
                        quantity_stock_final_prop_of_descripcion,
                        quantity_stock_final_prop_of_product,
                        quantity_stock_final_prop_of_customer,

                        -- Cumulative standardization stats
                        quantity_cust_request_qty_cumulative_mean,
                        quantity_cust_request_qty_cumulative_std,
                        quantity_cust_request_tn_cumulative_mean,
                        quantity_cust_request_tn_cumulative_std,
                        quantity_tn_cumulative_mean,
                        quantity_tn_cumulative_std,
                        quantity_stock_final_cumulative_mean,
                        quantity_stock_final_cumulative_std,
                            
                        -- Standardized product customer quantities
                        quantity_cust_request_qty_standardized,
                        quantity_cust_request_tn_standardized,
                        quantity_tn_standardized,
                        quantity_stock_final_standardized,
                        quantity_tn_target_standardized - quantity_tn_standardized AS target,
                            
                        -- Z Lags
                        {self._generate_lag_features_on_standardized_values(
                            z_mean_column="quantity_cust_request_qty_cumulative_mean",
                            z_std_column="quantity_cust_request_qty_cumulative_std",
                            column_name='quantity_cust_request_qty',
                            max_z_lag=self.dataset["max_lag_periods"],
                            partition_columns=['product_id', 'customer_id']
                        )},
                        {self._generate_lag_features_on_standardized_values(
                            z_mean_column="quantity_cust_request_tn_cumulative_mean",
                            z_std_column="quantity_cust_request_tn_cumulative_std",
                            column_name='quantity_cust_request_tn',
                            max_z_lag=self.dataset["max_lag_periods"],
                            partition_columns=['product_id', 'customer_id']
                        )},
                        {self._generate_lag_features_on_standardized_values(
                            z_mean_column="quantity_tn_cumulative_mean",
                            z_std_column="quantity_tn_cumulative_std",
                            column_name='quantity_tn',
                            max_z_lag=self.dataset["max_lag_periods"],
                            partition_columns=['product_id', 'customer_id']
                        )},
                        {self._generate_lag_features_on_standardized_values(
                            z_mean_column="quantity_stock_final_cumulative_mean",
                            z_std_column="quantity_stock_final_cumulative_std",
                            column_name='quantity_stock_final',
                            max_z_lag=self.dataset["max_lag_periods"],
                            partition_columns=['product_id', 'customer_id']
                        )},

                        -- Anchored Ratios
                        {self._generate_anchored_ratio_features(
                            column_name='quantity_cust_request_qty',
                            ratio_lag_periods=self.dataset["anchored_ratios_periods"],
                            partition_columns=['product_id', 'customer_id']
                        )},
                        {self._generate_anchored_ratio_features(
                            column_name='quantity_cust_request_tn',
                            ratio_lag_periods=self.dataset["anchored_ratios_periods"],
                            partition_columns=['product_id', 'customer_id']
                        )},
                        {self._generate_anchored_ratio_features(
                            column_name='quantity_tn',
                            ratio_lag_periods=self.dataset["anchored_ratios_periods"],
                            partition_columns=['product_id', 'customer_id']
                        )},
                        {self._generate_anchored_ratio_features(
                            column_name='quantity_stock_final',
                            ratio_lag_periods=self.dataset["anchored_ratios_periods"],
                            partition_columns=['product_id', 'customer_id']
                        )},

                        -- Rolling Stats
                        {self._generate_rolling_statistics(
                            column_name='quantity_cust_request_qty',
                            rolling_stats_window_sizes=self.dataset["rolling_stats_window_sizes"],
                            rolling_statistics=self.dataset["rolling_statistics"],
                            partition_columns=['product_id', 'customer_id']
                        )},
                        {self._generate_rolling_statistics(
                            column_name='quantity_cust_request_tn',
                            rolling_stats_window_sizes=self.dataset["rolling_stats_window_sizes"],
                            rolling_statistics=self.dataset["rolling_statistics"],
                            partition_columns=['product_id', 'customer_id']
                        )},
                        {self._generate_rolling_statistics(
                            column_name='quantity_tn',
                            rolling_stats_window_sizes=self.dataset["rolling_stats_window_sizes"],
                            rolling_statistics=self.dataset["rolling_statistics"],
                            partition_columns=['product_id', 'customer_id']
                        )},
                        {self._generate_rolling_statistics(
                            column_name='quantity_stock_final',
                            rolling_stats_window_sizes=self.dataset["rolling_stats_window_sizes"],
                            rolling_statistics=self.dataset["rolling_statistics"],
                            partition_columns=['product_id', 'customer_id']
                        )},

                        -- Regression Slopes
                        {self._generate_regression_slopes(
                            column_name='quantity_cust_request_qty',
                            regression_slopes_window_sizes=self.dataset["reg_slopes_window_sizes"],
                            partition_columns=['product_id', 'customer_id']
                        )},
                        {self._generate_regression_slopes(
                            column_name='quantity_cust_request_tn',
                            regression_slopes_window_sizes=self.dataset["reg_slopes_window_sizes"],
                            partition_columns=['product_id', 'customer_id']
                        )},
                        {self._generate_regression_slopes(
                            column_name='quantity_tn',
                            regression_slopes_window_sizes=self.dataset["reg_slopes_window_sizes"],
                            partition_columns=['product_id', 'customer_id']
                        )},
                        {self._generate_regression_slopes(
                            column_name='quantity_stock_final',
                            regression_slopes_window_sizes=self.dataset["reg_slopes_window_sizes"],
                            partition_columns=['product_id', 'customer_id']
                        )}
                        
                    FROM raw_and_standardized_product_customer_quantities
                )
                SELECT *,

                    -- Anchored Delta Z-lag
                    {self._generate_anchored_delta_features_on_zlag_columns(
                        base_column_name='quantity_cust_request_qty',
                        max_z_lag=self.dataset["max_z_lag_periods"],
                        z_anchored_delta_lag_periods=self.dataset["z_anchored_delta_lag_periods"]
                    )},
                    {self._generate_anchored_delta_features_on_zlag_columns(
                        base_column_name='quantity_cust_request_tn',
                        max_z_lag=self.dataset["max_z_lag_periods"],
                        z_anchored_delta_lag_periods=self.dataset["z_anchored_delta_lag_periods"]
                    )},
                    {self._generate_anchored_delta_features_on_zlag_columns(
                        base_column_name='quantity_tn',
                        max_z_lag=self.dataset["max_z_lag_periods"],
                        z_anchored_delta_lag_periods=self.dataset["z_anchored_delta_lag_periods"]
                    )},
                    {self._generate_anchored_delta_features_on_zlag_columns(
                        base_column_name='quantity_stock_final',
                        max_z_lag=self.dataset["max_z_lag_periods"],
                        z_anchored_delta_lag_periods=self.dataset["z_anchored_delta_lag_periods"]
                    )},

                    -- Adjacent Delta Z-lag
                    {self._generate_adjacent_delta_features_on_zlag_columns(
                        base_column_name='quantity_cust_request_qty',
                        max_z_lag=self.dataset["max_z_lag_periods"],
                        z_adjacent_delta_lag_periods=self.dataset["z_adjacent_delta_lag_periods"]
                    )},
                    {self._generate_adjacent_delta_features_on_zlag_columns(
                        base_column_name='quantity_cust_request_tn',
                        max_z_lag=self.dataset["max_z_lag_periods"],
                        z_adjacent_delta_lag_periods=self.dataset["z_adjacent_delta_lag_periods"]
                    )},
                    {self._generate_adjacent_delta_features_on_zlag_columns(
                        base_column_name='quantity_tn',
                        max_z_lag=self.dataset["max_z_lag_periods"],
                        z_adjacent_delta_lag_periods=self.dataset["z_adjacent_delta_lag_periods"]
                    )},
                    {self._generate_adjacent_delta_features_on_zlag_columns(
                        base_column_name='quantity_stock_final',
                        max_z_lag=self.dataset["max_z_lag_periods"],
                        z_adjacent_delta_lag_periods=self.dataset["z_adjacent_delta_lag_periods"]
                    )},

                    -- Anchored Z-lag ratios
                    {self._generate_anchored_ratio_features_on_zlag_columns(
                        base_column_name='quantity_cust_request_qty',
                        max_z_lag=self.dataset["max_z_lag_periods"],
                        z_anchored_ratio_lag_periods=self.dataset["z_anchored_ratio_lag_periods"]
                    )},
                    {self._generate_anchored_ratio_features_on_zlag_columns(
                        base_column_name='quantity_cust_request_tn',
                        max_z_lag=self.dataset["max_z_lag_periods"],
                        z_anchored_ratio_lag_periods=self.dataset["z_anchored_ratio_lag_periods"]
                    )},
                    {self._generate_anchored_ratio_features_on_zlag_columns(
                        base_column_name='quantity_tn',
                        max_z_lag=self.dataset["max_z_lag_periods"],
                        z_anchored_ratio_lag_periods=self.dataset["z_anchored_ratio_lag_periods"]
                    )},
                    {self._generate_anchored_ratio_features_on_zlag_columns(
                        base_column_name='quantity_stock_final',
                        max_z_lag=self.dataset["max_z_lag_periods"],
                        z_anchored_ratio_lag_periods=self.dataset["z_anchored_ratio_lag_periods"]
                    )},

                    -- Adjacent Z-lag ratios
                    {self._generate_adjacent_ratio_features_on_zlag_columns(
                        base_column_name='quantity_cust_request_qty',
                        max_z_lag=self.dataset["max_z_lag_periods"],
                        z_adjacent_ratio_lag_periods=self.dataset["z_adjacent_ratio_lag_periods"]
                    )},
                    {self._generate_adjacent_ratio_features_on_zlag_columns(
                        base_column_name='quantity_cust_request_tn',
                        max_z_lag=self.dataset["max_z_lag_periods"],
                        z_adjacent_ratio_lag_periods=self.dataset["z_adjacent_ratio_lag_periods"]
                    )},
                    {self._generate_adjacent_ratio_features_on_zlag_columns(
                        base_column_name='quantity_tn',
                        max_z_lag=self.dataset["max_z_lag_periods"],
                        z_adjacent_ratio_lag_periods=self.dataset["z_adjacent_ratio_lag_periods"]
                    )},
                    {self._generate_adjacent_ratio_features_on_zlag_columns(
                        base_column_name='quantity_stock_final',
                        max_z_lag=self.dataset["max_z_lag_periods"],
                        z_adjacent_ratio_lag_periods=self.dataset["z_adjacent_ratio_lag_periods"]
                    )},

                    -- Z-lag regression slopes
                    {self._generate_regression_slopes_on_zlag_columns(
                        base_column_name='quantity_cust_request_qty',
                        max_z_lag=self.dataset["max_z_lag_periods"],
                        z_regression_slopes_window_sizes=self.dataset["z_adjacent_ratio_lag_periods"]
                    )},
                    {self._generate_regression_slopes_on_zlag_columns(
                        base_column_name='quantity_cust_request_tn',
                        max_z_lag=self.dataset["max_z_lag_periods"],
                        z_regression_slopes_window_sizes=self.dataset["z_adjacent_ratio_lag_periods"]
                    )},
                    {self._generate_regression_slopes_on_zlag_columns(
                        base_column_name='quantity_tn',
                        max_z_lag=self.dataset["max_z_lag_periods"],
                        z_regression_slopes_window_sizes=self.dataset["z_adjacent_ratio_lag_periods"]
                    )},
                    {self._generate_regression_slopes_on_zlag_columns(
                        base_column_name='quantity_stock_final',
                        max_z_lag=self.dataset["max_z_lag_periods"],
                        z_regression_slopes_window_sizes=self.dataset["z_adjacent_ratio_lag_periods"]
                    )},

                FROM fixed_standardized_lag_series
                WHERE periodo >= {self.dataset["min_periodo"]}  -- Only {self.dataset["min_periodo"]} and onwards
                ORDER BY product_id, customer_id, periodo
            );             
            """)
            
            temp_path = "/tmp/dataset.parquet"
            self.conn.execute(f"COPY dataset TO '{temp_path}' (FORMAT PARQUET)")
            
            logger.info(f"Logging dataset to MLflow at {self.dataset['dataset_name']}...")

            mlflow.log_artifact(
                temp_path,
                artifact_path=self.dataset["dataset_name"]
            )

            os.remove(temp_path)

            logger.info(f"Dataset generated and logged to MLflow at {self.dataset['dataset_name']}")
            
            # Log dataset metadata
            mlflow.log_param("dataset_len", str(self.conn.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]))
            mlflow.log_param("dataset_columns", str(self.conn.execute("DESCRIBE dataset").fetchall()))

            # Tag the dataset as generated
            mlflow.set_tag("dataset_generated", "true")
