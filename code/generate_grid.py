import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
from torch_geometric.utils import degree

# 📍 File path
file_path = '/projects/kzhou6/bcui2/git_submit/SWE_25/data/all_points_final_merged_training.csv'
chunksize = 500000

# 📋 Intended columns
base_columns = [
    'date', 'lat', 'lon', 'Elevation', 'Slope', 'Aspect', 'Curvature', 'Northness', 'Eastness',
    'SWE', 'fsca', 'air_temperature_tmmx', 'air_temperature_tmmn',
    'potential_evapotranspiration', 'relative_humidity_rmax', 'relative_humidity_rmin',
    'mean_vapor_pressure_deficit', 'wind_speed', 'snodas_mask', 'water_year', 'swe_value'
]

time_series_cols = [
    f"{col}_{i}" for col in [
        "SWE", "air_temperature_tmmx", "air_temperature_tmmn", "potential_evapotranspiration",
        "relative_humidity_rmax", "relative_humidity_rmin", "mean_vapor_pressure_deficit",
        "precipitation_amount", "wind_speed", "fsca"
    ] for i in range(1, 8)
]

# cumulative_cols = [
#     "cumulative_SWE", "cumulative_air_temperature_tmmn", "cumulative_air_temperature_tmmx",
#     "cumulative_potential_evapotranspiration", "cumulative_mean_vapor_pressure_deficit",
#     "cumulative_precipitation_amount", "cumulative_relative_humidity_rmax",
#     "cumulative_relative_humidity_rmin", "cumulative_wind_speed", "cumulative_fsca"
# ]

intended_columns = base_columns + time_series_cols #+ cumulative_cols

# 🔍 Check available columns
print("Checking available columns in CSV...")
preview = pd.read_csv(file_path, nrows=1)
available_columns = set(preview.columns)
useful_columns = [col for col in intended_columns if col in available_columns]
if 'swe_value' not in useful_columns:
    useful_columns.append('swe_value')

missing_columns = [col for col in intended_columns if col not in available_columns]
if missing_columns:
    print(f"Missing columns skipped: {missing_columns}")

# 📥 Load dataset
print("\n Loading dataset...")
df_list = []
for chunk in pd.read_csv(file_path, usecols=useful_columns, chunksize=chunksize):
    if 'date' in chunk.columns:
        chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
        chunk = chunk.dropna(subset=['date'])
        chunk['day_of_year'] = chunk['date'].dt.dayofyear
    else:
        chunk['day_of_year'] = 1

    if 'swe_value' in chunk.columns:
        chunk = chunk[(chunk['swe_value'] >= 0) & (chunk['swe_value'] < 3000)]

    df_list.append(chunk)

data = pd.concat(df_list, ignore_index=True)

data_drop = data.drop_duplicates()

# 🧮 Binning
grid_size = 0.5
num_time_bins = 8
data['lat_bin'] = (data['lat'] // grid_size).astype(int)
data['lon_bin'] = (data['lon'] // grid_size).astype(int)
data['dayofyear_bin'] = (data['day_of_year'] // (365 / num_time_bins)).astype(int)
data['grid_id'] = data['lat_bin'].astype(str) + "_" + data['lon_bin'].astype(str) + "_" + data['dayofyear_bin'].astype(str)

data.to_csv('/groups/ESS/whung/swe_gnn/data/data_grid_id.csv', index=False)

#import pandas as pd
#
#df = pd.read_csv('/projects/kzhou6/bcui2/git_submit/SWE_25/data/data_grid_id.csv')
#
#print(df['grid_id'].nunique())
#
#print(df['grid_id'].value_counts())
