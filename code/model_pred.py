
import numpy as np
import pandas as pd
import xarray as xr

from eval_util import plot_model_products, plot_model_scatter

noaa_file_path = '/groups/ESS/whung/swe_gnn/data/noaa_snodas'
arizona_file_path = '/groups/ESS/whung/swe_gnn/data/swe_uarizona'
obs_file = '/groups/ESS/whung/swe_gnn/data/all_snotel_cdec_stations_active_in_westus_2025-01-01_2025-09-06.csv'
saved_file = '/groups/ESS/whung/swe_gnn/data/snotel_cdec_stations_model_paired.csv'
target_dates = ['2025-01-01', '2025-01-08', '2025-01-15', '2025-01-22', '2025-01-29', '2025-02-05', '2025-02-12', '2025-02-19', '2025-02-26']

def meter_to_inch(data):
    return data * 39.37

def noaa_swe_reader(filename):
    readin = xr.open_dataset(filename)
    lon = readin.lon.data
    lat = readin.lat.data
    data = readin.Band1.data
    data = data / 1000   # scale factor = 1000
    data = meter_to_inch(data)   
    readin.close()
    return {'lon': lon, 'lat': lat, 'swe': data}

def arizona_swe_reader(filename):
    readin = xr.open_dataset(filename)
    lon = readin.lon.data
    lat = readin.lat.data
    data = readin.SWE.data[0, :, :]
    data = data / 1000   # mm -> m
    data = meter_to_inch(data)   
    readin.close()
    return {'lon': lon, 'lat': lat, 'swe': data}

def site_pairing(obs_data, model_data):
    site_lon = site_data['lon']
    site_lat = site_data['lat']

    model_lon = model_data['lon']
    model_lat = model_data['lat']
    model_swe = model_data['swe']

    paired_swe = []
    xx, yy = np.meshgrid(model_lon, model_lat)
    for x, y in zip(site_lon, site_lat):
        dis = (xx - x)**2 + (yy - y)**2
        swe = np.nanmean(model_swe[dis == np.min(dis)])
        paired_swe = np.append(paired_swe, swe)
    paired_swe[np.isnan(paired_swe)] = 0.0  # avoid NaNs
    return paired_swe


obs_data = pd.read_csv(obs_file)

all_lon_paired = []
all_lat_paired = []
all_obs_paired = []
all_noaa_paired = []
all_ua_paired = []

for date in target_dates:
    d = pd.to_datetime(date, format='%Y-%m-%d')
    d = d.strftime('%Y%m%d')

    ## model products
    noaa_file = f"{noaa_file_path}/zz_ssmv11034tS__T0001TTNATS{d}05HP001.nc"
    arizona_file = f"{arizona_file_path}/UA_SWE_Depth_4km_v1_{d}_stable.nc"

    print(f'\n---- Reading model prodcucts {date}...')
    print(noaa_file)
    print(arizona_file)

    noaa_prod = noaa_swe_reader(noaa_file)
    arizona_prod = arizona_swe_reader(arizona_file)

    ## site observation
    print(f'\n---- Reading site observation {date}...')
    date_idx = np.squeeze(np.argwhere((obs_data['date']==date) & (~np.isnan(obs_data['swe_value']))))   # select data on the desired date with no NaN
    obs_id  = obs_data['station_name'][date_idx]
    obs_lon = np.round(obs_data['lon'][date_idx], 3)
    obs_lat = np.round(obs_data['lat'][date_idx], 3)
    obs_swe = obs_data['swe_value'][date_idx]
    site_data = {
        'id': obs_id,
        'lon': obs_lon,
        'lat': obs_lat,
        'swe': obs_swe
    }
    print('Num of site observation:', len(obs_swe))

    print('---- Site pairing...')
    noaa_swe = site_pairing(site_data, noaa_prod)
    ua_swe = site_pairing(site_data, arizona_prod)
    print(f"Num of paired sites: (NOAA) {np.count_nonzero(~np.isnan(noaa_swe))} (Arizona) {np.count_nonzero(~np.isnan(ua_swe))}")

    print('---- Plotting daily maps...')

    #plot_model_products(
    #    noaa_prod,
    #    'NOAA',
    #    date,
    #    noaa_file_path
    #)

    #plot_model_products(
    #    arizona_prod,
    #    'Arizona',
    #    date,
    #    arizona_file_path
    #)

    all_lon_paired = np.append(all_lon_paired, obs_lon)
    all_lat_paired = np.append(all_lat_paired, obs_lat)
    all_obs_paired = np.append(all_obs_paired, obs_swe)
    all_noaa_paired = np.append(all_noaa_paired, noaa_swe)
    all_ua_paired = np.append(all_ua_paired, ua_swe)

all_paired_data = {
    'lon': all_lon_paired,
    'lat': all_lat_paired,
    'swe_obs': all_obs_paired,
    'swe_noaa': all_noaa_paired,
    'swe_ua': all_ua_paired
}

print('\n---- Saving to csv file...')
all_paired_df = pd.DataFrame(data=all_paired_data)
all_paired_df.to_csv(saved_file, index=False)
print(f'Saved to {saved_file}')

print('---- Plotting density scatter plots...')
plot_model_scatter(
    all_obs_paired,
    all_noaa_paired,
    'NOAA',
    noaa_file_path
)

plot_model_scatter(
    all_obs_paired,
    all_ua_paired,
    'Arizona',
    arizona_file_path
)
print('Plots generated!')
