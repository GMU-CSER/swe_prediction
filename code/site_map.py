import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

path = '/groups/ESS/whung/swe_gnn/data'
train_file = f'{path}/all_points_final_merged_training_snodas_mask_resnet_all_batch.csv'
test_file = f'{path}/all_snotel_cdec_stations_active_in_westus_2025-01-01_2025-09-06.csv'
shp_USA = '/groups/ESS/whung/shp_files/gadm36_USA_shp/gadm36_USA_1'

## domain setting
lon_lim = [-127, -98]
lon_ticks = np.arange(-125, -100+1, 5)
lat_lim = [30, 50]
lat_ticks = np.arange(30, 50+1, 5)
print(f"Lon={lon_lim}, Lat={lat_lim}")

## read train sites (including fake zero-swe sites)
train_site = pd.read_csv(train_file, usecols=['lon', 'lat', 'station_name', 'swe_value'])
train_site.loc[train_site['swe_value'] < 0, 'swe_value'] = np.nan
train_site = train_site.dropna()
train_site.loc[train_site['station_name'] == '0', 'station_name'] = 0

fake_site = train_site.loc[(train_site.station_name==0) & (train_site.swe_value==0)]
fake_site = fake_site.groupby(['lon', 'lat']).agg(station_name=('station_name', 'mean'), swe=('swe_value', 'mean')).reset_index()

train_site = train_site.loc[train_site.station_name!=0]
train_site = train_site.groupby(['station_name']).agg(lon=('lon', 'mean'), lat=('lat', 'mean'), swe=('swe_value', 'mean')).reset_index()
sitelist = np.sort(train_site.station_name).tolist()

ghcn = []
for item in sitelist:
    if ('US' in item) or ('CA' in item):
        ghcn.append(item)
ghcn_site = train_site[train_site['station_name'].isin(ghcn)]

snotel = [x for x in train_site.station_name if x not in ghcn]
snotel_site = train_site[train_site['station_name'].isin(snotel)]

print('\n--------------------------------------')
print(f'Train site number={len(sitelist)}')
print(f'GHCN site number={ghcn_site.shape[0]}')
print(ghcn_site)
print(f'SNOTEL site number={snotel_site.shape[0]}')
print(snotel_site)
print(f'Fake site number={fake_site.shape[0]}')
print(fake_site)

## read test sites (actual SNOTEL sites)
test_site = pd.read_csv(test_file, usecols=['lon', 'lat', 'station_name', 'date', 'swe_value'])
test_site['date'] = pd.to_datetime(test_site['date'])
date_idx = (test_site['date'].dt.dayofyear < pd.to_datetime('2025-03-01').dayofyear) & (~np.isnan(test_site['swe_value']))
test_site = test_site.loc[date_idx].round(4)
test_site = test_site.groupby(['station_name']).agg(lon=('lon', 'mean'), lat=('lat', 'mean')).reset_index()
print('\n--------------------------------------')
print(f'Test site number={test_site.shape[0]}')
print(test_site)

## plot site map for training
fig, ax = plt.subplots(figsize=(15, 12))    # unit=100pixel
h = ax.get_position()
ax.set_position([h.x0-0.03, h.y0, h.width+0.01, h.height])
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)

plt.title(f'2018 - 2021 SWE sites\n(N_real={len(sitelist)}, N_random={fake_site.shape[0]})', fontsize=24)

m = Basemap(llcrnrlon=lon_lim[0],urcrnrlon=lon_lim[-1],llcrnrlat=lat_lim[0],urcrnrlat=lat_lim[-1], projection='cyl', resolution='l')
#m.arcgisimage(service='NGS_Topo_US_2D', xpixels=2000, verbose=True)
m.arcgisimage(service='World_Physical_Map', xpixels=2000, verbose=True)
m.readshapefile(shp_USA, 'USA', color='k')
m.drawmeridians(lon_ticks, color='none', labels=[0,0,0,1], fontsize=24)
m.drawparallels(lat_ticks, color='none', labels=[1,0,0,0], fontsize=24)

x, y = m(fake_site['lon'], fake_site['lat'])
m.scatter(x, y, marker='o', c='w', s=50, edgecolor='k', alpha=0.8, label='Random')
x, y = m(ghcn_site['lon'], ghcn_site['lat'])
m.scatter(x, y, marker='o', c='skyblue', s=50, edgecolor='k', alpha=0.8, label='GHCN')
x, y = m(snotel_site['lon'], snotel_site['lat'])
m.scatter(x, y, marker='o', c='crimson', s=50, edgecolor='k', alpha=0.8, label='SNOTEL')

plt.legend(loc='lower left', prop={'size':20})

plt.savefig(f'{path}/site_map.png')
plt.close()


## plot site map for evaluation
fig, ax = plt.subplots(figsize=(15, 12))    # unit=100pixel
h = ax.get_position()
ax.set_position([h.x0-0.03, h.y0, h.width+0.01, h.height])
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)

plt.title(f'Winter 2025 SNOTEL sites', fontsize=24)

m = Basemap(llcrnrlon=lon_lim[0],urcrnrlon=lon_lim[-1],llcrnrlat=lat_lim[0],urcrnrlat=lat_lim[-1], projection='cyl', resolution='l')
#m.arcgisimage(service='NGS_Topo_US_2D', xpixels=2000, verbose=True)
m.arcgisimage(service='World_Physical_Map', xpixels=2000, verbose=True)
m.readshapefile(shp_USA, 'USA', color='k')
m.drawmeridians(lon_ticks, color='none', labels=[0,0,0,1], fontsize=24)
m.drawparallels(lat_ticks, color='none', labels=[1,0,0,0], fontsize=24)

x, y = m(test_site['lon'], test_site['lat'])
m.scatter(x, y, marker='o', c='crimson', s=60, edgecolor='k', label='SNOTEL')

plt.savefig(f'{path}/site_map_evaluation.png')
plt.close()