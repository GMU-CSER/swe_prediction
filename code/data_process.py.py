import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
from torch_geometric.utils import degree
import pickle
from torch_geometric.data import Data
from sklearn.metrics import mean_squared_error, r2_score

train_file_path = '/groups/ESS/whung/swe_gnn/data/all_points_final_merged_training_snodas_mask_resnet_all_batch.csv'
test_file_path = '/groups/ESS/whung/swe_gnn/data/testing_snodas_mask/testing_all_ready_2025-02-26_merged.csv_snodas_mask.csv'
#test_file_path = '/groups/ESS/whung/swe_gnn/data/test_data_predicted_latest_2025-01-15.csv_snodas_mask.csv'

chunksize = 500000

# 📋 Intended columns

intended_columns = ['lat', 'lon', 'date', 'precipitation_amount', 'relative_humidity_rmin', 'potential_evapotranspiration', 
                    'air_temperature_tmmx', 'relative_humidity_rmax', 'mean_vapor_pressure_deficit',
                    'air_temperature_tmmn', 'wind_speed', 'Elevation', 'Aspect', 'Curvature', 'Northness', 
                    'Eastness', 'fsca', 'Slope', 'air_temperature_tmmn_1', 'potential_evapotranspiration_1',
                    'mean_vapor_pressure_deficit_1', 'relative_humidity_rmax_1', 'relative_humidity_rmin_1', 
                    'precipitation_amount_1', 'air_temperature_tmmx_1', 'wind_speed_1', 'fsca_1',                     'air_temperature_tmmn_2', 'potential_evapotranspiration_2', 'mean_vapor_pressure_deficit_2', 
                    'relative_humidity_rmax_2', 'relative_humidity_rmin_2', 'precipitation_amount_2',
                    'air_temperature_tmmx_2', 'wind_speed_2', 'fsca_2', 'air_temperature_tmmn_3', 
                    'potential_evapotranspiration_3', 'mean_vapor_pressure_deficit_3', 'relative_humidity_rmax_3',
                    'relative_humidity_rmin_3', 'precipitation_amount_3', 'air_temperature_tmmx_3', 'wind_speed_3', 'fsca_3',
                    'air_temperature_tmmn_4', 'potential_evapotranspiration_4', 'mean_vapor_pressure_deficit_4', 
                    'relative_humidity_rmax_4', 'relative_humidity_rmin_4', 'precipitation_amount_4', 'air_temperature_tmmx_4', 
                    'wind_speed_4', 'fsca_4',
                    'air_temperature_tmmn_5', 'potential_evapotranspiration_5', 'mean_vapor_pressure_deficit_5', 
                    'relative_humidity_rmax_5', 'relative_humidity_rmin_5', 'precipitation_amount_5',
                    'air_temperature_tmmx_5', 'wind_speed_5', 'fsca_5', 'air_temperature_tmmn_6', 
                    'potential_evapotranspiration_6', 'mean_vapor_pressure_deficit_6', 'relative_humidity_rmax_6',
                    'relative_humidity_rmin_6', 'precipitation_amount_6', 'air_temperature_tmmx_6', 'wind_speed_6', 
                    'fsca_6', 'air_temperature_tmmn_7', 'potential_evapotranspiration_7',
                    'mean_vapor_pressure_deficit_7', 'relative_humidity_rmax_7', 'relative_humidity_rmin_7', 
                    'precipitation_amount_7', 'air_temperature_tmmx_7', 'wind_speed_7', 'fsca_7', 'water_year', 'snodas_mask']

# 🔍 Check available columns
print("Checking available columns in train CSV...")
preview_train = pd.read_csv(train_file_path, nrows=1)
print("Checking available columns in test CSV...")
preview_test = pd.read_csv(test_file_path, nrows=1)
print(f'The total features are ({len(preview_train.columns)}): {preview_train.columns}')
available_columns = set(preview_train.columns)
useful_columns = [col for col in intended_columns if col in available_columns]
if 'swe_value' not in useful_columns:
    useful_columns.append('swe_value')
print(f'The useful columns length is: {len(useful_columns)}')


missing_columns = [col for col in intended_columns if col not in available_columns]
if missing_columns:
    print(f"Missing columns skipped: {missing_columns}")

#select_columns = list(set(useful_columns) & set(preview_test.columns)) #+ ['date']
select_columns = list(set(useful_columns))
if 'swe_value' not in select_columns:
    select_columns.append('swe_value')
    print("Appending swe_value to select_columns")
print(f'The selected columns are ({len(select_columns)}): {select_columns}')


# 📥 Load dataset
print("\n Loading train dataset...")
train_df_list = []
for chunk in pd.read_csv(train_file_path, usecols=select_columns, chunksize=chunksize):
    if 'date' in chunk.columns:
        chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
        chunk = chunk.dropna(subset=['date'])
        chunk['day_of_year'] = chunk['date'].dt.dayofyear
    else:
        chunk['day_of_year'] = 1

    if 'swe_value' in chunk.columns:
        chunk = chunk[(chunk['swe_value'] >= 0) & (chunk['swe_value'] < 3000)]

    train_df_list.append(chunk)

train_data = pd.concat(train_df_list, ignore_index=True)

print("\n Loading testing dataset...")
test_data_temp = pd.read_csv(test_file_path)
#test_data_temp.rename(columns={'date_x': 'date'}, inplace=True)
#test_data_temp.drop(columns=['date_y'], inplace=True)
test_data_temp.dropna(subset=['date'], inplace=True)
if 'swe_value' not in test_data_temp.columns:
    test_data_temp = test_data_temp.assign(swe_value=pd.NA)

test_data = test_data_temp[select_columns]

if 'date' in test_data.columns:
    test_data['date'] = pd.to_datetime(test_data['date'], errors='coerce')
    test_data['day_of_year'] = test_data['date'].dt.dayofyear
else:
    test_data['day_of_year'] = 1

# 🧮 Binning
grid_size = 0.01
num_time_bins = 1

train_data['lat_bin'] = (train_data['lat'] // grid_size).astype(int)
train_data['lon_bin'] = (train_data['lon'] // grid_size).astype(int)
train_data['grid_id'] = train_data['lat_bin'].astype(str) + "_" + train_data['lon_bin'].astype(str) + "_" + train_data['day_of_year'].astype(str)

test_data['lat_bin'] = (test_data['lat'] // grid_size).astype(int)
test_data['lon_bin'] = (test_data['lon'] // grid_size).astype(int)
test_data['grid_id'] = test_data['lat_bin'].astype(str) + "_" + test_data['lon_bin'].astype(str) + "_" + test_data['day_of_year'].astype(str)

# 📦 Aggregation
agg_cols = {col: 'mean' for col in select_columns if col not in ['date', 'lat', 'lon']}
agg_cols.update({'lat': 'mean', 'lon': 'mean', 'day_of_year': 'mean'})
#agg_cols.update({'lat': 'mean', 'lon': 'mean'})
if 'date' in select_columns:
    agg_cols['date'] = 'first'
agg_cols['swe_value'] = 'mean'

train_merged_nodes = train_data.groupby('grid_id').agg(agg_cols).reset_index()

test_merged_nodes = test_data.groupby('grid_id').agg(agg_cols).reset_index()

# ⏳ Temporal encoding
train_merged_nodes['sin_day'] = np.sin(2 * np.pi * train_merged_nodes['day_of_year'] / 365)
train_merged_nodes['cos_day'] = np.cos(2 * np.pi * train_merged_nodes['day_of_year'] / 365)

test_merged_nodes['sin_day'] = np.sin(2 * np.pi * test_merged_nodes['day_of_year'] / 365)
test_merged_nodes['cos_day'] = np.cos(2 * np.pi * test_merged_nodes['day_of_year'] / 365)

# 🌍 Spatial encoding
train_merged_nodes['lat_rad'] = np.radians(train_merged_nodes['lat'])
train_merged_nodes['lon_rad'] = np.radians(train_merged_nodes['lon'])
train_merged_nodes['sin_lat'] = np.sin(train_merged_nodes['lat_rad'])
train_merged_nodes['cos_lat'] = np.cos(train_merged_nodes['lat_rad'])
train_merged_nodes['sin_lon'] = np.sin(train_merged_nodes['lon_rad'])
train_merged_nodes['cos_lon'] = np.cos(train_merged_nodes['lon_rad'])

test_merged_nodes['lat_rad'] = np.radians(test_merged_nodes['lat'])
test_merged_nodes['lon_rad'] = np.radians(test_merged_nodes['lon'])
test_merged_nodes['sin_lat'] = np.sin(test_merged_nodes['lat_rad'])
test_merged_nodes['cos_lat'] = np.cos(test_merged_nodes['lat_rad'])
test_merged_nodes['sin_lon'] = np.sin(test_merged_nodes['lon_rad'])
test_merged_nodes['cos_lon'] = np.cos(test_merged_nodes['lon_rad'])

# 🧪 Final feature selection
exclude = ['grid_id', 'lat', 'lon', 'lat_rad', 'lon_rad', 'date', 'day_of_year', 'swe_value']
final_columns = [col for col in train_merged_nodes.columns if col not in exclude]
final_columns = np.sort(final_columns).tolist()   # make sure columns always have the same order 
#final_columns += ['sin_lat', 'cos_lat', 'sin_lon', 'cos_lon', 'sin_day', 'cos_day']
print(f'The final columns are ({len(final_columns)}): {final_columns}')

# ⚙️ Normalize
scaler = StandardScaler()
train_node_features_scaled = scaler.fit_transform(train_merged_nodes[final_columns])
train_df_scaled = pd.DataFrame(train_node_features_scaled, columns=final_columns)
scaler_sava_path = '/groups/ESS/whung/swe_gnn/data/scaler.pkl'
with open(scaler_sava_path, 'wb') as f:
    pickle.dump(scaler, f)

test_node_features_scaled = scaler.transform(test_merged_nodes[final_columns])
test_df_scaled = pd.DataFrame(test_node_features_scaled, columns=final_columns)

# ⚙️ Filter correlations
#corr = train_df_scaled.corr().abs()
#upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
#to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
#print(f"Dropping highly correlated features: {to_drop}")
#train_df_scaled.drop(columns=to_drop, inplace=True)
#print(f"The remaining features are ({len(train_df_scaled.columns)}): {train_df_scaled.columns}")

#test_df_scaled.drop(columns=to_drop, inplace=True)
#print(f"The remaining features are ({len(test_df_scaled.columns)}): {test_df_scaled.columns}")

# Convert to tensor
def build_graph(merged_nodes, df_scaled, is_test=False):
    node_features = torch.tensor(df_scaled.values, dtype=torch.float)
    if not is_test:
        labels = torch.tensor(merged_nodes['swe_value'].values, dtype=torch.float)
    else:
        labels = None

    # 🌐 Build edges with KDTree
    coordinates = merged_nodes[['lat', 'lon']].values
    tree = KDTree(coordinates)
    k = 6
    threshold = 0.5
    distances, neighbors = tree.query(coordinates, k=k+1)

    edge_list = []
    for i in range(len(neighbors)):
        for j in range(1, k+1):
            if distances[i][j] < threshold:
                edge_list.append((i, neighbors[i][j]))

    edges = np.array(edge_list).T
    edge_index = torch.tensor(edges, dtype=torch.long) if edges.size > 0 else torch.empty((2, 0), dtype=torch.long)

    # 🧵 Create Graph
    if not is_test:
        graph_data = Data(x=node_features, edge_index=edge_index, y=labels)
    else:
        graph_data = Data(x=node_features, edge_index=edge_index)
    # 保存 grid_id 便于预测结果映射
    graph_data.grid_id = merged_nodes['grid_id'].tolist()
    graph_data.dates = merged_nodes['date'].astype(str).tolist() if 'date' in merged_nodes.columns else []

    # 📊 Graph Summary
    print("\n Graph Summary")
    print(f"🔹 Nodes: {graph_data.num_nodes}")
    print(f"🔹 Edges: {graph_data.num_edges}")
    try:
        print(f"🔹 grid_id: {graph_data.grid_id}")
    except:
        print("🔹 grid_id: not found")
    print(f"🔹 Features per node: {graph_data.num_node_features}")
    deg = degree(edge_index[0], num_nodes=graph_data.num_nodes)
    print(f"   • Min Degree : {deg.min().item()}")
    print(f"   • Max Degree : {deg.max().item()}")
    print(f"   • Mean Degree: {deg.float().mean():.2f}")
    print(f"   • Isolated   : {(deg == 0).sum().item()} nodes")

    return graph_data

#print("-----------------------<<< Building training graph >>>-----------------------\n")
#train_graph_data = build_graph(train_merged_nodes, train_df_scaled, is_test=False)
print("\n-----------------------<<< Building test graph >>>-----------------------\n")
test_graph_data = build_graph(test_merged_nodes, test_df_scaled, is_test=True)

# 💾 Save graph
#train_save_path = '/groups/ESS/whung/swe_gnn/data/gnn_training_data.pt'
#torch.save(train_graph_data, train_save_path)
#print(f"\n Graph data saved at: {train_save_path}")

test_save_path = '/groups/ESS/whung/swe_gnn/data/gnn_testing_data_2025-02-26.pt'
torch.save(test_graph_data, test_save_path)
print(f"\n Graph data saved at: {test_save_path}")