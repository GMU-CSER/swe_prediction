import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from eval_util import stats_calculation, plot_prediction_map, plot_scatter, plot_elv_r2_rmse, plot_stats_map

import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

obs_file = '/groups/ESS/whung/swe_gnn/data/all_snotel_cdec_stations_active_in_westus_2025-01-01_2025-09-06.csv'
pred_path = '/groups/ESS/whung/swe_gnn/data/testing_snodas_mask'
#pred_path = '/groups/ESS/whung/swe_gnn/data/testing_predicted'
target_dates = ['2025-01-01', '2025-01-08', '2025-01-15', '2025-01-22', '2025-01-29', '2025-02-05', '2025-02-12', '2025-02-19', '2025-02-26']

print('------ Reading data...')

all_obs = []
all_id = []
all_lat = []
all_lon = []
all_elv = []
all_fsca = []
all_pred_mseloss = []
all_pred_sweloss = []
all_pred_rmseloss = []
all_pred_et = []

obs_data = pd.read_csv(obs_file)

for date in target_dates:
    print(f'\n------ {date}')

    ## site observation
    date_idx = np.squeeze(np.argwhere((obs_data['date']==date) & (~np.isnan(obs_data['swe_value']))))   # select data on the desired date with no NaN
    obs_id  = obs_data['station_name'][date_idx]
    obs_lon = np.round(obs_data['lon'][date_idx], 3)
    obs_lat = np.round(obs_data['lat'][date_idx], 3)
    obs_swe = obs_data['swe_value'][date_idx]
    print('Num of site observation:', len(obs_swe))

    ## swe prediction
    pred_data = pd.read_csv(f'{pred_path}/testing_all_ready_{date}_merged.csv_snodas_mask_pred.csv')
    #pred_data = pd.read_csv(f'{pred_path}/testing_all_ready_{date}_merged.csv_snodas_mask_pred_v2.csv')
    #pred_data = pd.read_csv(f'{pred_path}/test_data_predicted_latest_{date}.csv_snodas_mask.csv')
    pred_lon = np.round(pred_data['lon'], 3)
    pred_lat = np.round(pred_data['lat'], 3)

    #plot_prediction_map(pred_data, 'ETHole', date, '/groups/ESS/whung/swe_gnn/results_et')

    pred_elv = []
    pred_fsca = []
    pred_swe_mseloss = []
    pred_swe_sweloss = []
    pred_swe_rmseloss = []
    pred_swe_et = []
    for x, y, n in zip(obs_lon, obs_lat, obs_id):
        dis = (pred_lon-x)**2 + (pred_lat-y)**2
        idx = np.squeeze(np.argwhere(dis==np.min(dis)))
        
        if np.sqrt(np.mean(dis[idx])) > 0.01:   # only use paired prediction within 1km
            pred_elv = np.append(pred_elv, np.nan)
            pred_fsca = np.append(pred_fsca, np.nan)
            pred_swe_mseloss = np.append(pred_swe_mseloss, np.nan)
            pred_swe_sweloss = np.append(pred_swe_sweloss, np.nan)
            pred_swe_rmseloss = np.append(pred_swe_rmseloss, np.nan)
            #pred_swe_et = np.append(pred_swe_et, np.nan)
        else:
            pred_elv = np.append(pred_elv, np.mean(pred_data['Elevation'][idx]))
            pred_fsca = np.append(pred_fsca, np.mean(pred_data['fsca'][idx]))
            pred_swe_mseloss = np.append(pred_swe_mseloss, np.mean(pred_data['predicted_swe_GCN_MSELoss'][idx]))
            pred_swe_sweloss = np.append(pred_swe_sweloss, np.mean(pred_data['predicted_swe_GCN_SWELoss'][idx]))
            pred_swe_rmseloss = np.append(pred_swe_rmseloss, np.mean(pred_data['predicted_swe_GCN_RMSELoss'][idx]))
            #pred_swe_et = np.append(pred_swe_et, np.mean(pred_data['predicted_swe'][idx]))
    print('Num of paired prediction:', len(np.argwhere(~np.isnan(pred_swe_mseloss))))
    #print('Num of paired prediction:', len(np.argwhere(~np.isnan(pred_swe_et))))

    all_obs = np.append(all_obs, obs_swe)
    all_id  = np.append(all_id, obs_id)
    all_lon = np.append(all_lon, obs_lon)
    all_lat = np.append(all_lat, obs_lat)
    all_elv = np.append(all_elv, pred_elv)
    all_fsca = np.append(all_fsca, pred_fsca)
    all_pred_mseloss = np.append(all_pred_mseloss, pred_swe_mseloss)
    all_pred_sweloss = np.append(all_pred_sweloss, pred_swe_sweloss)
    all_pred_rmseloss = np.append(all_pred_rmseloss, pred_swe_rmseloss)
    #all_pred_et = np.append(all_pred_et, pred_swe_et)

    del [date_idx, obs_lon, obs_lat, obs_id, obs_swe]
    del [pred_data, dis, idx, pred_lon, pred_lat, pred_elv, pred_fsca, pred_swe_mseloss, pred_swe_sweloss, pred_swe_rmseloss, pred_swe_et]

all_data = pd.DataFrame(data={
    'obs_swe': all_obs,
    'obs_fsca': all_fsca, 
    'site_id': all_id, 
    'site_lon': all_lon,
    'site_lat': all_lat,
    'site_elv': all_elv,
    'pred_swe_mseloss': all_pred_mseloss,
    'pred_swe_sweloss': all_pred_sweloss,
    'pred_swe_rmseloss': all_pred_rmseloss
    #'pred_swe_et': all_pred_et
})
all_data = all_data.dropna()
print('\n------ Final paired data:')
print('Num of data points:', len(all_data))

plot_scatter(
    all_data['obs_swe'],
    all_data['pred_swe_mseloss'],
    all_data['site_elv'],
    'GCN_MSELoss',
    '/groups/ESS/whung/swe_gnn/results_mseloss',
)
plot_scatter(
    all_data['obs_swe'],
    all_data['pred_swe_sweloss'],
    all_data['site_elv'],
    'GCN_SWELoss',
    '/groups/ESS/whung/swe_gnn/results_sweloss',
)
plot_scatter(
    all_data['obs_swe'],
    all_data['pred_swe_rmseloss'],
    all_data['site_elv'],
    'GCN_RMSELoss',
    '/groups/ESS/whung/swe_gnn/results_rmseloss',
)
#plot_scatter(
#    all_data['obs_swe'],
#    all_data['pred_swe_et'],
#    all_data['site_elv'],
#    'ETHole',
#    '/groups/ESS/whung/swe_gnn/results_et',
#)


## Site based analysis
print('\n------------------------------------')
print('\n------ Site based analysis')

sitelist = all_data.groupby(['site_id']).agg(
    site_lon=('site_lon', 'mean'),
    site_lat=('site_lat', 'mean'),
    site_elv=('site_elv', 'mean')
).reset_index()
print('Selected sites: num =', len(sitelist))
print(sitelist)

r2_sites = {'mseloss': [], 'sweloss': [], 'rmseloss': [], 'et': []}
rmse_sites = {'mseloss': [], 'sweloss': [], 'rmseloss': [], 'et': []}
for site_id, lon, lat in zip(sitelist['site_id'], sitelist['site_lon'], sitelist['site_lat']):
    X = all_data['obs_swe'][all_data['site_id']==site_id]
    Y1 = all_data['pred_swe_mseloss'][all_data['site_id']==site_id]
    Y2 = all_data['pred_swe_sweloss'][all_data['site_id']==site_id]
    Y3 = all_data['pred_swe_rmseloss'][all_data['site_id']==site_id]
    #Y3 = all_data['pred_swe_et'][all_data['site_id']==site_id]

    R2_1, RMSE_1 = stats_calculation(X, Y1)
    R2_2, RMSE_2 = stats_calculation(X, Y2)
    R2_3, RMSE_3 = stats_calculation(X, Y3)

    print('* Site name:', site_id)
    print('  Site location:', lat, lon)
    print('  R2= (mseloss)', R2_1, '(sweloss)', R2_2, '(rmseloss)', R2_3)
    print('  RMSE= (mseloss)', RMSE_1, '(sweloss)', RMSE_2, '(rmseloss)', RMSE_3)
    #print('  R2=', R2_3)
    #print('  RMSE=', RMSE_3)

    r2_sites['mseloss'] = np.append(r2_sites['mseloss'], R2_1)
    r2_sites['sweloss'] = np.append(r2_sites['sweloss'], R2_2)
    r2_sites['rmseloss'] = np.append(r2_sites['rmseloss'], R2_3)
    rmse_sites['mseloss'] = np.append(rmse_sites['mseloss'], RMSE_1)
    rmse_sites['sweloss'] = np.append(rmse_sites['sweloss'], RMSE_2)
    rmse_sites['rmseloss'] = np.append(rmse_sites['rmseloss'], RMSE_3)
    #r2_sites['et'] = np.append(r2_sites['et'], R2_3)
    #rmse_sites['et'] = np.append(rmse_sites['et'], RMSE_3)
    del [X, Y1, Y2, R2_1, R2_2, RMSE_1, RMSE_2, Y3, R2_3, RMSE_3]

print('\n------ Creating statistics table...')
stats = sitelist
stats['r2_GCN_MSELoss'] = r2_sites['mseloss']
stats['rmse_GCN_MSELoss'] = rmse_sites['mseloss']
stats['r2_GCN_SWELoss'] = r2_sites['sweloss']
stats['rmse_GCN_SWELoss'] = rmse_sites['sweloss']
stats['r2_GCN_RMSELoss'] = r2_sites['rmseloss']
stats['rmse_GCN_RMSELoss'] = rmse_sites['rmseloss']
#stats['r2_ETHole'] = r2_sites['et']
#stats['rmse_ETHole'] = rmse_sites['et']
print(stats)

print('\n------ Saving table as csv file...')
stats.to_csv('/groups/ESS/whung/swe_gnn/data/all_snotel_cdec_stations_prediction_site_stats.csv', index=False)
#stats.to_csv('/groups/ESS/whung/swe_gnn/data/all_snotel_cdec_stations_prediction_site_stats_et.csv', index=False)
print('------ Table saved!')

print('\n------ Plotting stats maps...')
plot_stats_map(stats, 'GCN_MSELoss', '/groups/ESS/whung/swe_gnn/results_mseloss')
plot_stats_map(stats, 'GCN_SWELoss', '/groups/ESS/whung/swe_gnn/results_sweloss')
plot_stats_map(stats, 'GCN_RMSELoss', '/groups/ESS/whung/swe_gnn/results_rmseloss')
#plot_stats_map(stats, 'ETHole', '/groups/ESS/whung/swe_gnn/results_et')
print('------ Map saved!')

print('\n------ Plotting elv-r2-rmse plots...')
plot_elv_r2_rmse(stats, 'GCN_MSELoss', '/groups/ESS/whung/swe_gnn/results_mseloss')
plot_elv_r2_rmse(stats, 'GCN_SWELoss', '/groups/ESS/whung/swe_gnn/results_sweloss')
plot_elv_r2_rmse(stats, 'GCN_RMSELoss', '/groups/ESS/whung/swe_gnn/results_rmseloss')
#plot_elv_r2_rmse(stats, 'ETHole', '/groups/ESS/whung/swe_gnn/results_et')
print('------ Plot saved!')



## Terrain (elevation) based analysis
print('\n------------------------------------')
print('\n------ Elevation based analysis')

elv_bins = [0, 500, 1000, 2000, 3000, 4000]   # 5 bins
bin_index = np.digitize(all_data['site_elv'], elv_bins)

num_bins = []
r2_bins = {'mseloss': [], 'sweloss': [], 'rmseloss': [], 'et': []}
rmse_bins = {'mseloss': [], 'sweloss': [], 'rmseloss': [], 'et': []}
for bin_id in range(1, 6):
    X = all_data['obs_swe'][bin_index == bin_id]
    Y1 = all_data['pred_swe_mseloss'][bin_index == bin_id]
    Y2 = all_data['pred_swe_sweloss'][bin_index == bin_id]
    Y3 = all_data['pred_swe_rmseloss'][bin_index == bin_id]
    #Y3 = all_data['pred_swe_et'][bin_index == bin_id]

    R2_1, RMSE_1 = stats_calculation(X, Y1)
    R2_2, RMSE_2 = stats_calculation(X, Y2)
    R2_3, RMSE_3 = stats_calculation(X, Y3)

    print('* Elevation:', elv_bins[bin_id-1], '-', elv_bins[bin_id])
    print('  Num of paired data:', len(X))
    print('  R2= (mseloss)', R2_1, '(sweloss)', R2_2, '(rmseloss)', R2_3)
    print('  RMSE= (mseloss)', RMSE_1, '(sweloss)', RMSE_2, '(rmseloss)', RMSE_3)
    #print('  R2=', R2_3)
    #print('  RMSE=', RMSE_3)

    num_bins = np.append(num_bins, len(X))
    r2_bins['mseloss'] = np.append(r2_bins['mseloss'], R2_1)
    r2_bins['sweloss'] = np.append(r2_bins['sweloss'], R2_2)
    r2_bins['rmseloss'] = np.append(r2_bins['rmseloss'], R2_3)
    rmse_bins['mseloss'] = np.append(rmse_bins['mseloss'], RMSE_1)
    rmse_bins['sweloss'] = np.append(rmse_bins['sweloss'], RMSE_2)
    rmse_bins['rmseloss'] = np.append(rmse_bins['rmseloss'], RMSE_3)
    #r2_bins['et'] = np.append(r2_bins['et'], R2_3)
    #rmse_bins['et'] = np.append(rmse_bins['et'], RMSE_3)
    del [X, Y1, Y2, R2_1, R2_2, RMSE_1, RMSE_2, Y3, R2_3, RMSE_3]

print('\n------ Creating statistics table...')
stats = pd.DataFrame(data={
    'elevation_min': elv_bins[:5],
    'elevation_max': elv_bins[1:],
    'counts': num_bins.astype(int),
    'r2_GCN_MSELoss': r2_bins['mseloss'],
    'rmse_GCN_MSELoss': rmse_bins['mseloss'],
    'r2_GCN_SWELoss': r2_bins['sweloss'],
    'rmse_GCN_SWELoss': rmse_bins['sweloss'],
    'r2_GCN_RMSELoss': r2_bins['rmseloss'],
    'rmse_GCN_RMSELoss': rmse_bins['rmseloss']
    #'r2_ETHole': r2_bins['et'],
    #'rmse_ETHole': rmse_bins['et']
})
print(stats)

print('\n------ Saving table as csv file...')
stats.to_csv('/groups/ESS/whung/swe_gnn/data/all_snotel_cdec_stations_prediction_elevation_stats.csv', index=False)
#stats.to_csv('/groups/ESS/whung/swe_gnn/data/all_snotel_cdec_stations_prediction_elevation_stats_et.csv', index=False)
print('------ Table saved!')
