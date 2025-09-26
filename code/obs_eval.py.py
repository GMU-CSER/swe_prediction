import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

obs_file = '/groups/ESS/whung/swe_gnn/data/all_snotel_cdec_stations_active_in_westus_2025-01-01_2025-09-06.csv'
pred_path = '/groups/ESS/whung/swe_gnn/data/testing_snodas_mask'
target_dates = ['2025-01-01', '2025-01-08', '2025-01-15', '2025-01-22', '2025-01-29', '2025-02-05', '2025-02-12', '2025-02-19', '2025-02-26']

coast = gpd.read_file('/groups/ESS/whung/shp_files/ne_10m_coastline/ne_10m_coastline.shp')

def stats_calculation(X, Y):
    R2 = np.corrcoef(X, Y)[0, 1]
    RMSE = np.sqrt(np.mean((Y-X)**2))
    return R2, RMSE

def plot_scatter(X, Y, C, model_name, pic_path):
    r2, rmse = stats_calculation(X, Y)

    tick_limit = np.ceil(np.max([X.max(), Y.max()]))

    plt.figure(figsize=(10, 10))
    plt.plot(np.arange(tick_limit),
        np.arange(tick_limit),
        linewidth=1,
        linestyle=':',
        color='k'
    )
    cs = plt.scatter(X, Y, c=C, s=30, cmap='turbo', vmin=500, vmax=2000, alpha=0.8)
    plt.annotate(
        text='R2='+('%.4f'%r2)+', RMSE='+('%.4f'%rmse),
        xy=(0.1, 0.9),
        xycoords='figure fraction',
        fontsize=20
    )

    plt.title(f'Result Comparison ({model_name})')
    plt.xlabel('Observation')
    plt.ylabel(f'Model: {model_name}')
    plt.xlim([0, tick_limit])
    plt.ylim([0, tick_limit])
    plt.grid(True, axis='y')

    cb = plt.colorbar(cs, extend='both')
    cb.set_label('Elevation')
    
    plt.tight_layout()
    plt.savefig(f'{pic_path}/obs_site_comparison.png')
    plt.close()

def plot_stats_map(data, model_option, pic_path):
    lat = np.array(data['site_lat'])
    lon = np.array(data['site_lon'])
    stats = {
        'R2': np.array(data[f'r2_{model_option}']),
        'RMSE': np.array(data[f'rmse_{model_option}'])
    }

    for item in stats:
        if item == 'R2':
            vlim = [-1, 1]
            cmap = 'seismic'
            cb_ext = 'neither'
        elif item == 'RMSE':
            vlim = [0, 20]
            cmap = 'turbo'
            cb_ext = 'max'

        fig, ax = plt.subplots(figsize=(15, 7))
        h = ax.get_position()
        ax.set_position([h.x0-0.04, h.y0+0.06, h.width+0.06, h.height])
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
        ax.tick_params(labelsize=24)

        plt.title(f'{item} of SWE Prediction from {model_option} Model', fontsize=24)
        coast.plot(ax=ax, linewidth=1, color='k')

        cs = plt.scatter(lon, lat, c=stats[item], s=100, marker='.', cmap=cmap, vmin=vlim[0], vmax=vlim[1])

        plt.xlim([-130, -60])
        plt.xticks(np.arange(-130, -60+1, 10))
        plt.ylim([25, 50])
        plt.yticks(np.arange(25, 50+1, 5))
        plt.grid(linewidth=1, linestyle=':')

        cbax = fig.add_axes([h.x0-0.04, h.y0+0.03, h.width+0.06, 0.02])
        cb   = plt.colorbar(cs, extend=cb_ext, orientation='horizontal', cax=cbax)
        #cb.set_ticks()
        cb.set_label(item, fontsize=24, fontweight='bold')
        cb.outline.set_linewidth(3)
        cb.ax.tick_params(labelsize=24)

        plt.savefig(f'{pic_path}/site_stats_{item}.png')
        plt.close()

def plot_elv_r2_rmse(data, model_option, pic_path):
    elv = data['site_elv']
    r2 = data[f'r2_{model_option}']
    rmse = data[f'rmse_{model_option}']

    plt.figure(figsize=(10, 10))
    cs = plt.scatter(elv, r2, c=rmse, s=30, cmap='turbo', vmin=0, vmax=20, alpha=0.8)

    plt.title(f'Elevation-R2-RMSE ({model_name})')
    plt.xlabel('Elevation')
    plt.ylabel('R2')
    plt.xlim([0, 4000])
    plt.ylim([-1, 1])

    cb = plt.colorbar(cs, extend='max')
    cb.set_label('RMSE')
    
    plt.tight_layout()
    plt.savefig(f'{pic_path}/elv_r2_rmse_comparison.png')
    plt.close()

print('------ Reading data...')

all_obs = []
all_id = []
all_lat = []
all_lon = []
all_elv = []
all_pred_mseloss = []
all_pred_sweloss = []

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
    pred_lon = np.round(pred_data['lon'], 3)
    pred_lat = np.round(pred_data['lat'], 3)

    pred_elv = []
    pred_swe_mseloss = []
    pred_swe_sweloss = []
    for x, y, n in zip(obs_lon, obs_lat, obs_id):
        dis = (pred_lon-x)**2 + (pred_lat-y)**2
        idx = np.squeeze(np.argwhere(dis==np.min(dis)))
        
        if np.sqrt(np.mean(dis[idx])) > 0.01:   # only use paired prediction within 1km
            pred_elv = np.append(pred_elv, np.nan)
            pred_swe_mseloss = np.append(pred_swe_mseloss, np.nan)
            pred_swe_sweloss = np.append(pred_swe_sweloss, np.nan)
        else:
            pred_elv = np.append(pred_elv, np.mean(pred_data['Elevation'][idx]))
            pred_swe_mseloss = np.append(pred_swe_mseloss, np.mean(pred_data['predicted_swe_GCN_MSELoss'][idx]))
            pred_swe_sweloss = np.append(pred_swe_sweloss, np.mean(pred_data['predicted_swe_GCN_SWELoss'][idx]))
    print('Num of paired prediction:', len(np.argwhere(~np.isnan(pred_swe_mseloss))))

    all_obs = np.append(all_obs, obs_swe)
    all_id  = np.append(all_id, obs_id)
    all_lon = np.append(all_lon, obs_lon)
    all_lat = np.append(all_lat, obs_lat)
    all_elv = np.append(all_elv, pred_elv)
    all_pred_mseloss = np.append(all_pred_mseloss, pred_swe_mseloss)
    all_pred_sweloss = np.append(all_pred_sweloss, pred_swe_sweloss)

    del [date_idx, obs_lon, obs_lat, obs_id, obs_swe]
    del [pred_data, dis, idx, pred_lon, pred_lat, pred_swe_mseloss, pred_swe_sweloss, pred_elv]

all_data = pd.DataFrame(data={
    'obs_swe': all_obs,
    'site_id': all_id, 
    'site_lon': all_lon,
    'site_lat': all_lat,
    'site_elv': all_elv,
    'pred_swe_mseloss': all_pred_mseloss,
    'pred_swe_sweloss': all_pred_sweloss
})
all_data = all_data.dropna()
print('\n------ Final paired data:')
print('Num of data points:', len(all_data))
print(np.unique(all_data['site_id']))
print(all_data['site_elv'][all_data['site_id']=='Bateman'])
exit()

plot_scatter(
    all_data['obs_swe'],
    all_data['pred_swe_mseloss'],
    all_data['site_elv'],
    'GCN_MSELoss',
    '/groups/ESS/whung/swe_gnn/results_mseloss'
)
plot_scatter(
    all_data['obs_swe'],
    all_data['pred_swe_sweloss'],
    all_data['site_elv'],
    'GCN_SWELoss',
    '/groups/ESS/whung/swe_gnn/results_sweloss'
)


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

r2_sites = {'mseloss': [], 'sweloss': []}
rmse_sites = {'mseloss': [], 'sweloss': []}
for site_id, lon, lat in zip(sitelist['site_id'], sitelist['site_lon'], sitelist['site_lat']):
    X = all_data['obs_swe'][all_data['site_id']==site_id]
    Y1 = all_data['pred_swe_mseloss'][all_data['site_id']==site_id]
    Y2 = all_data['pred_swe_sweloss'][all_data['site_id']==site_id]

    R2_1, RMSE_1 = stats_calculation(X, Y1)
    R2_2, RMSE_2 = stats_calculation(X, Y2)

    print('* Site name:', site_id)
    print('  Site location:', lat, lon)
    print('  R2= (mseloss)', R2_1, '(sweloss)', R2_2)
    print('  RMSE= (mseloss)', RMSE_1, '(sweloss)', RMSE_2)

    r2_sites['mseloss'] = np.append(r2_sites['mseloss'], R2_1)
    r2_sites['sweloss'] = np.append(r2_sites['sweloss'], R2_2)
    rmse_sites['mseloss'] = np.append(rmse_sites['mseloss'], RMSE_1)
    rmse_sites['sweloss'] = np.append(rmse_sites['sweloss'], RMSE_2)
    del [X, Y1, Y2, R2_1, R2_2, RMSE_1, RMSE_2]

print('\n------ Creating statistics table...')
stats = sitelist
stats['r2_GCN_MSELoss'] = r2_sites['mseloss']
stats['rmse_GCN_MSELoss'] = rmse_sites['mseloss']
stats['r2_GCN_SWELoss'] = r2_sites['sweloss']
stats['rmse_GCN_SWELoss'] = rmse_sites['sweloss']
print(stats)

print('\n------ Saving table as csv file...')
stats.to_csv('/groups/ESS/whung/swe_gnn/data/all_snotel_cdec_stations_prediction_site_stats.csv', index=False)
print('------ Table saved!')

print('\n------ Plotting stats maps...')
plot_stats_map(stats, 'GCN_MSELoss', '/groups/ESS/whung/swe_gnn/results_mseloss')
plot_stats_map(stats, 'GCN_SWELoss', '/groups/ESS/whung/swe_gnn/results_sweloss')
print('------ Map saved!')


## Terrain (elevation) based analysis
print('\n------------------------------------')
print('\n------ Elevation based analysis')

elv_bins = [0, 500, 1000, 2000, 3000, 4000]   # 5 bins
bin_index = np.digitize(all_data['site_elv'], elv_bins)

num_bins = []
r2_bins = {'mseloss': [], 'sweloss': []}
rmse_bins = {'mseloss': [], 'sweloss': []}
for bin_id in range(1, 6):
    X = all_data['obs_swe'][bin_index == bin_id]
    Y1 = all_data['pred_swe_mseloss'][bin_index == bin_id]
    Y2 = all_data['pred_swe_sweloss'][bin_index == bin_id]

    R2_1, RMSE_1 = stats_calculation(X, Y1)
    R2_2, RMSE_2 = stats_calculation(X, Y2)

    print('* Elevation:', elv_bins[bin_id-1], '-', elv_bins[bin_id])
    print('  Num of paired data:', len(X))
    print('  R2= (mseloss)', R2_1, '(sweloss)', R2_2)
    print('  RMSE= (mseloss)', RMSE_1, '(sweloss)', RMSE_2)

    num_bins = np.append(num_bins, len(X))
    r2_bins['mseloss'] = np.append(r2_bins['mseloss'], R2_1)
    r2_bins['sweloss'] = np.append(r2_bins['sweloss'], R2_2)
    rmse_bins['mseloss'] = np.append(rmse_bins['mseloss'], RMSE_1)
    rmse_bins['sweloss'] = np.append(rmse_bins['sweloss'], RMSE_2)
    del [X, Y1, Y2, R2_1, R2_2, RMSE_1, RMSE_2]

print('\n------ Creating statistics table...')
stats = pd.DataFrame(data={
    'elevation_min': elv_bins[:5],
    'elevation_max': elv_bins[1:],
    'counts': num_bins.astype(int),
    'r2_GCN_MSELoss': r2_bins['mseloss'],
    'rmse_GCN_MSELoss': rmse_bins['mseloss'],
    'r2_GCN_SWELoss': r2_bins['sweloss'],
    'rmse_GCN_SWELoss': rmse_bins['sweloss']
})
print(stats)

print('\n------ Saving table as csv file...')
stats.to_csv('/groups/ESS/whung/swe_gnn/data/all_snotel_cdec_stations_prediction_elevation_stats.csv', index=False)
print('------ Table saved!')
