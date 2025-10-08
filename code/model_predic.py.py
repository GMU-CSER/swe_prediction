
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import torch, pickle
from model_train import GCN_Model
from torch_geometric.loader import NeighborLoader

from eval_util import plot_prediction_map

saved_model_mseloss = '/groups/ESS/whung/swe_gnn/model/GCN_model_mseloss.pth'
saved_model_sweloss = '/groups/ESS/whung/swe_gnn/model/GCN_model_sweloss.pth'
#saved_model_sweloss = '/groups/ESS/whung/swe_gnn/model/GCN_model_v2.pth'
saved_model_rmseloss = '/groups/ESS/whung/swe_gnn/model/GCN_model_rmseloss.pth'
test_file = '/groups/ESS/whung/swe_gnn/data/testing_snodas_mask/testing_all_ready_2025-01-15_merged.csv_snodas_mask.csv'
test_file_pt = '/groups/ESS/whung/swe_gnn/data/gnn_testing_data_2025-01-15.pt'

with open('/groups/ESS/whung/swe_gnn/data/scaler.pkl','rb') as f:
    scaler = pickle.load(f)

model_params = {
    'in_channels': 86,  # number of input columns
    'hidden_channels': 64,
    'out_channels': 1,
    'num_heads': 4,
    'K': 3
}

def make_prediction(model: torch.nn.Module, input_data: torch.tensor) -> np.float64:
    with torch.no_grad():
        pred = model(input_data)
    pred = pred.numpy()
    return pred

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    ## read test data
    print('---- Loading test data...')

    test_data = pd.read_csv(test_file)
    date = test_data['date'][0]
    print('\n---- Original csv data:')
    print(f"Loaded from: {test_file}")
    print(test_data.columns)

    test_data_pt = torch.load(test_file_pt, map_location=device, weights_only=False)
    # 打印详细的数据信息
    print("\n---- Graph data structure:")
    print(f"Loaded from: {test_file_pt}")
    print(f"Type: {type(test_data_pt)}")
    print("\nAttributes:")
    
    # 打印基本属性
    print("x:")
    print(f"  Type: {type(test_data_pt.x)}")
    print(f"  Shape: {test_data_pt.x.shape}")
    print(f"  Dtype: {test_data_pt.x.dtype}")
    print(f"  Device: {test_data_pt.x.device}")
    
    print("\nedge_index:")
    print(f"  Type: {type(test_data_pt.edge_index)}")
    print(f"  Shape: {test_data_pt.edge_index.shape}")
    print(f"  Dtype: {test_data_pt.edge_index.dtype}")
    print(f"  Device: {test_data_pt.edge_index.device}")

    # 打印其他属性
    print("\nOther attributes:")
    for key in ['num_nodes', 'num_edges', 'num_node_features']:
        if hasattr(test_data_pt, key):
            print(f"{key}: {getattr(test_data_pt, key)}")

    ## extract lat/lon/elv from graph (use cos_lat/cos_lon/Elevation)
    ## check "final column" in the data process script for indexing
    print('\n---- Extracting original lat/lon...')
    pt_value = test_data_pt.x.cpu().numpy()
    pt_value_ori = scaler.inverse_transform(pt_value)
    scaled_lon = pt_value_ori[:, 24]
    scaled_lat = pt_value_ori[:, 23]
    lon = -1 * np.rad2deg(np.arccos(scaled_lon))   # arccos always returns positvie values
    lat = np.rad2deg(np.arccos(scaled_lat))
    elv = pt_value_ori[:, 3]
    fsca = pt_value_ori[:, 25]
    print('Scaled lon:', scaled_lon)
    print('Scaled lat:', scaled_lat)
    print('Original lon:', lon)
    print('Original lat:', lat)
    print('Original elv:', elv)
    print('Original fsca:', fsca)

    ## read trained model and make prediction
    models = {
        'GCN_MSELoss': torch.load(saved_model_mseloss, map_location=device),
        'GCN_SWELoss': torch.load(saved_model_sweloss, map_location=device),
        'GCN_RMSELoss': torch.load(saved_model_rmseloss, map_location=device)
    }

    output = {}
    for model_name, model_dict in models.items():
        print(f"---- Loading {model_name} model...")
        model = GCN_Model(model_params['in_channels'], model_params['hidden_channels'], model_params['out_channels'])
        model.load_state_dict(model_dict)
        model.eval()
        print(f"Model structure:")
        print(model)

        print('---- Predicting...')
        test_data_pt = test_data_pt.to(device)
        prediction = make_prediction(model, test_data_pt)
        print('Data shape:', prediction.shape)
        print('Check NaNs:', np.argwhere(np.isnan(prediction)))
        
        output[f"predicted_swe_{model_name}"] = prediction

    print('\n---- All predictions')
    pred_data = pd.DataFrame(data={
        'lon': np.round(lon, 3),
        'lat': np.round(lat, 2),
        'Elevation': elv,
        'fsca': fsca,
        'date': np.repeat(date, len(lon))})
    for item in output:
        pred_data[item] = output[item]
    print(pred_data)

    print('\n---- Saving to csv file...')
    pred_data.to_csv(test_file[:-4]+'_pred.csv', index=False)
    #pred_data.to_csv(test_file[:-4]+'_pred_v2.csv', index=False)
    print('---- Prediction saved!')

    print('\n---- Plotting prediction maps...')
    plot_prediction_map(pred_data, 'GCN_MSELoss', pred_data['date'][0], '/groups/ESS/whung/swe_gnn/results_mseloss')
    plot_prediction_map(pred_data, 'GCN_SWELoss', pred_data['date'][0], '/groups/ESS/whung/swe_gnn/results_sweloss')
    #plot_prediction_map(pred_data, 'GCN_SWELoss', pred_data['date'][0], '/groups/ESS/whung/swe_gnn/results_v2')
    plot_prediction_map(pred_data, 'GCN_RMSELoss', pred_data['date'][0], '/groups/ESS/whung/swe_gnn/results_rmseloss')
    print('---- Map saved!')


if __name__ == "__main__":
    main()