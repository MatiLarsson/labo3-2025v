# Construccion del dataset de products
generator = SQLFeaturesGenerator()

# Crear tabla base SIN features generadas
conn.execute(f"""
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
        GROUP BY
            product_id
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
            AND ps.periodo <= LEAST(ca.customer_last_active_period, pa.product_last_active_period)
        GROUP BY
            dp.product_id, dc.customer_id, ps.periodo,
            p.cat1, p.cat2, p.cat3, p.brand, p.sku_size, p.descripcion
    ),
    product_customer_period_target AS (
        SELECT
            product_id,
            customer_id,
            periodo,
             
            -- Target column: quantity_tn_target (quantity_tn shifted by 2 periods forward)
            LEAD(quantity_tn, 2, 0) OVER (PARTITION BY product_id, customer_id ORDER BY periodo) AS quantity_tn_target
             
        FROM active_product_customer_period
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
            STDDEV(quantity_cust_request_qty) OVER (
                PARTITION BY product_id, customer_id
                ORDER BY periodo
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS quantity_cust_request_qty_cumulative_std,
                                
            -- Cumulative stats for quantity_cust_request_tn
            
            -- Cumulative Mean: For standardization (z-score calculation)
            AVG(quantity_cust_request_tn) OVER (
                PARTITION BY product_id, customer_id
                ORDER BY periodo
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS quantity_cust_request_tn_cumulative_mean,
            -- Cumulative Std: For standardization (z-score calculation)
            STDDEV(quantity_cust_request_tn) OVER (
                PARTITION BY product_id, customer_id
                ORDER BY periodo
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS quantity_cust_request_tn_cumulative_std,
                                
            -- Cumulative stats for quantity_tn
            
            -- Cumulative Mean: For standardization (z-score calculation)
            AVG(quantity_tn) OVER (
                PARTITION BY product_id, customer_id
                ORDER BY periodo
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS quantity_tn_cumulative_mean,
            -- Cumulative Std: For standardization (z-score calculation)
            STDDEV(quantity_tn) OVER (
                PARTITION BY product_id, customer_id
                ORDER BY periodo
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS quantity_tn_cumulative_std
                                
        FROM active_product_customer_period
    ),
    raw_and_standardized_product_customer_quantities AS (
        SELECT 
            apcp.product_id,
            apcp.customer_id,
            apcp.periodo,
            apcp.cat1, apcp.cat2, apcp.cat3, apcp.brand, apcp.sku_size, apcp.descripcion,
            apcp.dummy_plan_precios_cuidados,
            
            -- Standardization stats
            pcpcss.quantity_cust_request_qty_cumulative_mean,
            pcpcss.quantity_cust_request_qty_cumulative_std,
            pcpcss.quantity_cust_request_tn_cumulative_mean,
            pcpcss.quantity_cust_request_tn_cumulative_std,
            pcpcss.quantity_tn_cumulative_mean,
            pcpcss.quantity_tn_cumulative_std,
             
            -- Raw quantities
            apcp.quantity_cust_request_qty,
            apcp.quantity_cust_request_tn,
            apcp.quantity_tn, 
             
            -- Standardized quantities
            CASE 
                WHEN pcpcss.quantity_cust_request_qty_cumulative_std > 0 
                THEN (apcp.quantity_cust_request_qty - pcpcss.quantity_cust_request_qty_cumulative_mean) / pcpcss.quantity_cust_request_qty_cumulative_std
                ELSE 0 
            END AS quantity_cust_request_qty_standardized,
            
            CASE 
                WHEN pcpcss.quantity_cust_request_tn_cumulative_std > 0 
                THEN (apcp.quantity_cust_request_tn - pcpcss.quantity_cust_request_tn_cumulative_mean) / pcpcss.quantity_cust_request_tn_cumulative_std
                ELSE 0 
            END AS quantity_cust_request_tn_standardized,
            
            CASE 
                WHEN pcpcss.quantity_tn_cumulative_std > 0 
                THEN (apcp.quantity_tn - pcpcss.quantity_tn_cumulative_mean) / pcpcss.quantity_tn_cumulative_std
                ELSE 0
            END AS quantity_tn_standardized,
            
            CASE 
                WHEN pcpcss.quantity_tn_cumulative_std > 0 
                THEN (pcpt.quantity_tn_target - pcpcss.quantity_tn_cumulative_mean) / pcpcss.quantity_tn_cumulative_std
                ELSE 0
            END AS quantity_tn_target_standardized
            
        FROM active_product_customer_period apcp
        LEFT JOIN product_customer_period_cumulative_standardization_stats pcpcss ON apcp.product_id = pcpcss.product_id 
            AND apcp.customer_id = pcpcss.customer_id
            AND apcp.periodo = pcpcss.periodo
        LEFT JOIN product_customer_period_target pcpt ON apcp.product_id = pcpt.product_id
            AND apcp.customer_id = pcpt.customer_id
            AND apcp.periodo = pcpt.periodo
    ),
    fixed_standardized_lag_series AS (
        SELECT
            product_id,
            customer_id,
            periodo,
            cat1,
            cat2,
            cat3,
            brand,
            sku_size,
            descripcion,
                                    
            -- Dummy features
            dummy_plan_precios_cuidados,
                
            -- Raw quantities
            quantity_cust_request_qty,
            quantity_cust_request_tn,
            quantity_tn,
                
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
            quantity_tn_target_standardized,
             
            -- Rolling Stats
            {generator.generate_rolling_statistics(
                column_name='quantity_cust_request_qty',
                rolling_stats_window_sizes=[3, 5, 7, 9, 11],
                rolling_statistics=['mean', 'median', 'std', 'min', 'max', 'sum', 'count', 'variance', 'skewness', 'kurtosis'],
                partition_columns=['product_id', 'customer_id']
            )},
            {generator.generate_rolling_statistics(
                column_name='quantity_cust_request_tn',
                rolling_stats_window_sizes=[3, 5, 7, 9, 11],
                rolling_statistics=['mean', 'median', 'std', 'min', 'max', 'sum', 'count', 'variance', 'skewness', 'kurtosis'],
                partition_columns=['product_id', 'customer_id']
            )},
            {generator.generate_rolling_statistics(
                column_name='quantity_tn',
                rolling_stats_window_sizes=[3, 5, 7, 9, 11],
                rolling_statistics=['mean', 'median', 'std', 'min', 'max', 'sum', 'count', 'variance', 'skewness', 'kurtosis'],
                partition_columns=['product_id', 'customer_id']
            )},

            -- Deltas
            {generator.generate_delta_features(
                column_name='quantity_cust_request_qty',
                delta_lag_periods=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                partition_columns=['product_id', 'customer_id']
            )},
            {generator.generate_delta_features(
                column_name='quantity_cust_request_tn',
                delta_lag_periods=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                partition_columns=['product_id', 'customer_id']
            )},
            {generator.generate_delta_features(
                column_name='quantity_tn',
                delta_lag_periods=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                partition_columns=['product_id', 'customer_id']
            )},

            -- Ratios
            {generator.generate_ratio_features(
                column_name='quantity_cust_request_qty',
                ratio_lag_periods=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                partition_columns=['product_id', 'customer_id']
            )},
            {generator.generate_ratio_features(
                column_name='quantity_cust_request_tn',
                ratio_lag_periods=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                partition_columns=['product_id', 'customer_id']
            )},
            {generator.generate_ratio_features(
                column_name='quantity_tn',
                ratio_lag_periods=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                partition_columns=['product_id', 'customer_id']
            )},

            -- Regression Slopes
            {generator.generate_regression_slopes(
                column_name='quantity_cust_request_qty',
                regression_slopes_window_sizes=[3, 5, 7, 9, 11],
                partition_columns=['product_id', 'customer_id']
            )},
            {generator.generate_regression_slopes(
                column_name='quantity_cust_request_tn',
                regression_slopes_window_sizes=[3, 5, 7, 9, 11],
                partition_columns=['product_id', 'customer_id']
            )},
            {generator.generate_regression_slopes(
                column_name='quantity_tn',
                regression_slopes_window_sizes=[3, 5, 7, 9, 11],
                partition_columns=['product_id', 'customer_id']
            )},
                
            -- Z Lags
            {generator.generate_lag_features_on_standardized_values(
                z_mean_column="quantity_cust_request_qty_cumulative_mean",
                z_std_column="quantity_cust_request_qty_cumulative_std",
                column_name='quantity_cust_request_qty',
                max_z_lag=11,
                partition_columns=['product_id', 'customer_id']
            )},
            {generator.generate_lag_features_on_standardized_values(
                z_mean_column="quantity_cust_request_tn_cumulative_mean",
                z_std_column="quantity_cust_request_tn_cumulative_std",
                column_name='quantity_cust_request_tn',
                max_z_lag=11,
                partition_columns=['product_id', 'customer_id']
            )},
            {generator.generate_lag_features_on_standardized_values(
                z_mean_column="quantity_tn_cumulative_mean",
                z_std_column="quantity_tn_cumulative_std",
                column_name='quantity_tn',
                max_z_lag=11,
                partition_columns=['product_id', 'customer_id']
            )}
            
        FROM raw_and_standardized_product_customer_quantities
    )
    SELECT *,

        -- Delta Z-lag (max_z_lag must be the same as z_delta_lag_periods chosen)
        {generator.generate_delta_features_on_zlag_columns(
            base_column_name='quantity_cust_request_qty',
            max_z_lag=11,
            z_delta_lag_periods=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        )},
        {generator.generate_delta_features_on_zlag_columns(
            base_column_name='quantity_cust_request_tn',
            max_z_lag=11,
            z_delta_lag_periods=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        )},
        {generator.generate_delta_features_on_zlag_columns(
            base_column_name='quantity_tn',
            max_z_lag=11,
            z_delta_lag_periods=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        )},

        -- Z-lag ratios (max_z_lag must be the same as z_delta_lag_periods chosen)
        {generator.generate_ratio_features_on_zlag_columns(
            base_column_name='quantity_cust_request_qty',
            max_z_lag=11,
            z_ratio_lag_periods=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        )},
        {generator.generate_ratio_features_on_zlag_columns(
            base_column_name='quantity_cust_request_tn',
            max_z_lag=11,
            z_ratio_lag_periods=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        )},
        {generator.generate_ratio_features_on_zlag_columns(
            base_column_name='quantity_tn',
            max_z_lag=11,
            z_ratio_lag_periods=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        )},

        -- Z-lag regression slopes (max_z_lag must be the same as z_delta_lag_periods chosen)
        {generator.generate_regression_slopes_on_zlag_columns(
            base_column_name='quantity_cust_request_qty',
            max_z_lag=11,
            z_regression_slopes_window_sizes=[3, 5, 7, 9, 11]
        )},
        {generator.generate_regression_slopes_on_zlag_columns(
            base_column_name='quantity_cust_request_tn',
            max_z_lag=11,
            z_regression_slopes_window_sizes=[3, 5, 7, 9, 11]
        )},
        {generator.generate_regression_slopes_on_zlag_columns(
            base_column_name='quantity_tn',
            max_z_lag=11,
            z_regression_slopes_window_sizes=[3, 5, 7, 9, 11]
        )}

    FROM fixed_standardized_lag_series
    ORDER BY product_id, customer_id, periodo
);
""")
