
import numpy as np
import pandas as pd
import xarray as xr

from eval_util import plot_model_products

noaa_file_path = '/groups/ESS/whung/swe_gnn/data/noaa_snodas'
arizona_file_path = '/groups/ESS/whung/swe_gnn/data/swe_uarizona'
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


for date in target_dates:
    d = pd.to_datetime(date, format='%Y-%m-%d')
    d = d.strftime('%Y%m%d')

    noaa_file = f"{noaa_file_path}/zz_ssmv11034tS__T0001TTNATS{d}05HP001.nc"
    arizona_file = f"{arizona_file_path}/UA_SWE_Depth_4km_v1_{d}_stable.nc"

    print(f'\n---- Reading data {date}...')
    print(noaa_file)
    print(arizona_file)

    noaa_prod = noaa_swe_reader(noaa_file)
    arizona_prod = arizona_swe_reader(arizona_file)

    print('---- Plotting maps...')

    plot_model_products(
        noaa_prod,
        'NOAA',
        date,
        noaa_file_path
    )

    plot_model_products(
        arizona_prod,
        'Arizona',
        date,
        arizona_file_path
    )
