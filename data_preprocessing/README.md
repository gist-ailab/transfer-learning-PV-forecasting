# Data Preprocessing Steps

## Datasets

### DKASC_AliceSprings

(add contents soon...)

### GIST_dataset

(add contents soon...)

In this project, we process datasets from various sites to unify formats and remove anomalies for further analysis. The preprocessing involves two main steps:

1. `1_unify_format.py`
2. `2_drop_anomaly.py`

Additional information:

- **raw_info**: After running `1_unify_format.py`, we visualize correlations for each site and note the maximum and minimum values for each column.
- **processed_info**: After running `2_drop_anomaly.py`, we again visualize correlations for each site and note the maximum and minimum values for each column.

# Unifying Data Format

## `1_unify_format.py`

This script unifies the data format across different datasets. The standardized CSV files are stored in the `uniform_format` folder for each country.

- **Timestamp**: Data is resampled or aggregated to hourly intervals.
- **Active_Power**: Measured in kilowatts (kW).

  - For the UK dataset, `Active_Power` is calculated as the average of `P_GEN_MIN` and `P_GEN_MAX`:

    ```python
    Active_Power = (P_GEN_MIN + P_GEN_MAX) / 2
    ```

- **Wind_Speed**: Measured in meters per second (m/s).
- **Weather_Relative_Humidity**: Measured in percentage (%).
- **Temperature**: Measured in degrees Celsius (°C).
- **Global_Horizontal_Radiation**: Measured in watts per square meter (W/m²).

# Anomaly Removal

## `2_drop_anomaly.py`

### Process

- **Completing Timestamps**: If data for a specific timestamp is missing, we fill all columns (except the timestamp itself) with `NaN` values to ensure a full day's worth of data.

  ```python
  # Function to ensure all hours are present for each day
  def ensure_full_day_timestamps(df, timestamp_col='timestamp'):
      """
      This function ensures that each day has all 24 hours (00:00 to 23:00).
      If any hour is missing, it will log the missing timestamps and return the modified DataFrame.
      """
  ```

- **Anomaly Removal**:

  - We remove both common anomalies across all sites and site-specific anomalies.

- **Handling Missing Data**:

  - If there are two or more consecutive timestamps with `NaN` values, we remove the entire day.
  - If there is only one missing value, we apply linear interpolation to estimate it.

## List of Anomalies

### Common Anomalies

The following anomalies are checked and corrected across all datasets:

- **Global Horizontal Radiation (GHR) exceeds physical limits**:

  - If `Global_Horizontal_Radiation` > 2000 W/m², the value is considered an anomaly.

- **Temperature below realistic thresholds**:

  - If `Weather_Temperature_Celsius` < -10°C, the value is considered an anomaly.

- **Negative Wind Speed**:

  - If `Wind_Speed` < 0 m/s, convert these values to `NaN` instead of setting them to zero.

- **Relative Humidity outside physical bounds**:

  - If `Weather_Relative_Humidity` < 0% or `Weather_Relative_Humidity` > 100%, the value is considered an anomaly.

- **Mismatch between Power and Radiation**:

  - If `Normalized_Active_Power` = 0 and `Global_Horizontal_Radiation` > 200 W/m², the power value is considered an anomaly.
  - Conversely, if `Normalized_Active_Power` > 0.1 and `Global_Horizontal_Radiation` < 10 W/m², the power value is considered an anomaly.

- **Repeating Non-Zero Active Power Values**:

  - If a non-zero `Active_Power` value is repeated 10 or more times consecutively, these values are converted to `NaN`.

    - *Example*: Site 85 from February 6, 2009, to March 18, 2009.

- **Slightly Negative Normalized Active Power**:

  - If `Normalized_Active_Power` is between -0.01 and 0 (i.e., slightly negative), we convert it to 0.

### Site-Specific Anomalies

#### DKASC_AliceSprings

- **Site 67-Site_DKA-M8_A-Phase**:

  - Convert negative `Active_Power` values to positive.

- **Sites 85-Site_DKA-M7_A-Phase and 98-Site_DKA-M8_B-Phase_site19**:

  - The maximum `Active_Power` values deviate significantly from the general trend. We will leave these values as is for now.

- **Site 90-Site_DKA-M3_A-Phase**:

  - Instances where `Normalized_Active_Power` > 0.5 are considered anomalies.

#### DKASC_Yulara

- Only the common anomalies are applied to this dataset.

#### German

- **Sites DE_KN_industrial2_pv and DE_KN_residential3_pv**:

  - Instances where `Normalized_Active_Power` > 0.2 are considered anomalies.

#### GIST

- **Sites**:

  - `C10_Renewable-E-Bldg_feature`
  - `C11_GAIA`
  - `E03_GTI`
  - `E12_DormB`
  - `N01_Central-Library`
  - `N02_LG-Library`
  - `W11_Facility-Maintenance-Bldg`
  - `W13_Centeral-Storage_feature`

- For these sites, negative `Active_Power` values are converted to positive.

#### UK

- Only the common anomalies are applied to this dataset.

#### OEDI_Georgia and OEDI_California

- After applying common anomalies:

  - **Active Power**:

    - Sum the `Active_Power` values from each inverter to get the total active power for the site.

  - **Other Columns (e.g., GHR)**:

    - Calculate the average values for other measurements.

---