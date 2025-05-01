import pandas as pd
import numpy as np
import os

# Input CSV
input_csv = "/workspaces/air_quality_analysis_spark/output/combined_data_singlefile/part-00000-a5b8ffcc-67be-4ed5-ab70-61b4bbd3eaad-c000.csv"

# Main output directory
main_output_dir = "/workspaces/air_quality_analysis_spark/output_task2"
os.makedirs(main_output_dir, exist_ok=True)

# Read CSV
df = pd.read_csv(input_csv, parse_dates=["timestamp"])

# 1. Clean Data
df_clean = df.copy()
df_clean = df_clean[df_clean["pm25"] < 1000]
df_clean["temperature"] = np.where(df_clean["temperature"] > 60, np.nan, df_clean["temperature"])
df_clean["humidity"] = np.where((df_clean["humidity"] > 100) | (df_clean["humidity"] < 0), np.nan, df_clean["humidity"])
df_clean["pm25"].fillna(df_clean["pm25"].median(), inplace=True)
df_clean["temperature"].fillna(df_clean["temperature"].median(), inplace=True)
df_clean["humidity"].fillna(df_clean["humidity"].median(), inplace=True)

# Round numeric values to 2 decimals
df_clean = df_clean.round(2)

# Save Cleaned Data
clean_dir = os.path.join(main_output_dir, "cleaned_data")
os.makedirs(clean_dir, exist_ok=True)
df_clean.to_csv(os.path.join(clean_dir, "task2_cleaned_data.csv"), index=False)
df_clean.to_parquet(os.path.join(clean_dir, "task2_cleaned_data.parquet"))
open(os.path.join(clean_dir, "_SUCCESS"), "w").close()

# 2. Z-Score Normalization
df_norm = df_clean.copy()
for col in ["pm25", "temperature", "humidity"]:
    mean = df_norm[col].mean()
    std = df_norm[col].std()
    df_norm[f"{col}_zscore"] = ((df_norm[col] - mean) / std).round(2)

# Save Enhanced Data
enhanced_dir = os.path.join(main_output_dir, "enhanced_data")
os.makedirs(enhanced_dir, exist_ok=True)
df_norm.to_csv(os.path.join(enhanced_dir, "task2_enhanced_data.csv"), index=False)
df_norm.to_parquet(os.path.join(enhanced_dir, "task2_enhanced_data.parquet"))
open(os.path.join(enhanced_dir, "_SUCCESS"), "w").close()

# 3. Daily Aggregations
df_norm["date"] = df_norm["timestamp"].dt.date
daily = df_norm.groupby(["date", "region"]).agg({
    "pm25": "mean",
    "temperature": "mean",
    "humidity": "mean"
}).reset_index().round(2)

# Save Daily Aggregations
daily_dir = os.path.join(main_output_dir, "daily_aggregations")
os.makedirs(daily_dir, exist_ok=True)
daily.to_csv(os.path.join(daily_dir, "task2_daily_aggregations.csv"), index=False)
daily.to_parquet(os.path.join(daily_dir, "task2_daily_aggregations.parquet"))
open(os.path.join(daily_dir, "_SUCCESS"), "w").close()

# 4. Hourly Aggregations + Rolling/Lag/Rate Change
df_norm["hour"] = df_norm["timestamp"].dt.hour
hourly = df_norm.groupby(["date", "hour", "region"]).agg({
    "pm25": "mean",
    "temperature": "mean",
    "humidity": "mean"
}).reset_index().round(2)

# Rolling, Lag and Rate of Change
df_norm = df_norm.sort_values(["region", "timestamp"])
df_norm["pm25_rolling_avg_3"] = df_norm.groupby("region")["pm25"].transform(lambda x: x.rolling(3, min_periods=1).mean()).round(2)
df_norm["pm25_lag_1"] = df_norm.groupby("region")["pm25"].shift(1).round(2)
df_norm["pm25_rate_of_change"] = (df_norm["pm25"] - df_norm["pm25_lag_1"]).round(2)

# Save Hourly + Trends
hourly_dir = os.path.join(main_output_dir, "hourly_trends")
os.makedirs(hourly_dir, exist_ok=True)
df_norm.to_csv(os.path.join(hourly_dir, "task2_hourly_trends.csv"), index=False)
df_norm.to_parquet(os.path.join(hourly_dir, "task2_hourly_trends.parquet"))
open(os.path.join(hourly_dir, "_SUCCESS"), "w").close()

print("🎯 Task 2 - COMPLETED Successfully!")
print(f"All outputs organized inside ➡️ {main_output_dir}")