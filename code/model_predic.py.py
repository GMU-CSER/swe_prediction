
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import torch, pickle
from model_train import GCN_Model
from torch_geometric.loader import NeighborLoader

saved_model_mseloss = '/groups/ESS/whung/swe_gnn/model/GCN_model_mseloss.pth'
saved_model_sweloss = '/groups/ESS/whung/swe_gnn/model/GCN_model_sweloss.pth'
test_file = '/groups/ESS/whung/swe_gnn/data/testing_snodas_mask/testing_all_ready_2025-01-15_merged.csv_snodas_mask.csv'
test_file_pt = '/groups/ESS/whung/swe_gnn/data/gnn_testing_data_2025-01-15.pt'

with open('/groups/ESS/whung/swe_gnn/data/scaler.pkl','rb') as f:
    scaler = pickle.load(f)
coast = gpd.read_file('/groups/ESS/whung/shp_files/ne_10m_coastline/ne_10m_coastline.shp')

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

def plot_prediction_map(data, model_option, pic_path):
    date = data['date'][0]
    lat = np.array(data['lat'])
    lon = np.array(data['lon'])
    swe = np.array(data[f'predicted_swe_{model_option}'])
    swe[swe < 0] = 0

    fig, ax = plt.subplots(figsize=(15, 7))
    h = ax.get_position()
    ax.set_position([h.x0-0.04, h.y0+0.06, h.width+0.06, h.height])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(labelsize=24)

    plt.title(f'{date} SWE Prediction from {model_option} Model', fontsize=24)
    coast.plot(ax=ax, linewidth=1, color='k')

    cs = plt.scatter(lon, lat, c=swe, s=10, marker='.', cmap='turbo', vmin=0, vmax=10)

    plt.xlim([-130, -60])
    plt.xticks(np.arange(-130, -60+1, 10))
    plt.ylim([25, 50])
    plt.yticks(np.arange(25, 50+1, 5))
    plt.grid(linewidth=1, linestyle=':')

    cbax = fig.add_axes([h.x0-0.04, h.y0+0.03, h.width+0.06, 0.02])
    cb   = plt.colorbar(cs, extend='max', orientation='horizontal', cax=cbax)
    #cb.set_ticks()
    cb.set_label('SWE value', fontsize=24, fontweight='bold')
    cb.outline.set_linewidth(3)
    cb.ax.tick_params(labelsize=24)

    plt.savefig(f'{pic_path}/swe_prediction_{date}.png')
    plt.close()

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
    pt_value = test_data_pt.x.numpy()
    pt_value_ori = scaler.inverse_transform(pt_value)
    scaled_lon = pt_value_ori[:, 24]
    scaled_lat = pt_value_ori[:, 23]
    lon = -1 * np.rad2deg(np.arccos(scaled_lon))   # arccos always returns positvie values
    lat = np.rad2deg(np.arccos(scaled_lat))
    elv = pt_value_ori[:, 3]
    print('Scaled lon:', scaled_lon)
    print('Scaled lat:', scaled_lat)
    print('Original lon:', lon)
    print('Original lat:', lat)
    print('Original elv:', elv)

    ## read trained model and make prediction
    models = {
        'GCN_MSELoss': torch.load(saved_model_mseloss, map_location=device),
        'GCN_SWELoss': torch.load(saved_model_sweloss, map_location=device)
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
        'date': np.repeat(date, len(lon))})
    for item in output:
        pred_data[item] = output[item]
    print(pred_data)

    print('\n---- Saving to csv file...')
    pred_data.to_csv(test_file[:-4]+'_pred.csv', index=False)
    print('---- Prediction saved!')

    print('\n---- Plotting prediction maps...')
    plot_prediction_map(pred_data, 'GCN_MSELoss', '/groups/ESS/whung/swe_gnn/results_mseloss')
    plot_prediction_map(pred_data, 'GCN_SWELoss', '/groups/ESS/whung/swe_gnn/results_sweloss')
    print('---- Map saved!')


if __name__ == "__main__":
    main()