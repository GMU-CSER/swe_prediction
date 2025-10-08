
import numpy as np
import pandas as pd

data_dir = '/groups/ESS/whung/swe_gnn/data/testing_snodas_mask'
target_date = '2025-02-26'
target_date_dt = pd.to_datetime(target_date, format='%Y-%m-%d')

required_columns = ['SWE', 'air_temperature_tmmn', 'potential_evapotranspiration', 'mean_vapor_pressure_deficit', 'relative_humidity_rmax', 'relative_humidity_rmin', 'precipitation_amount', 'air_temperature_tmmx', 'wind_speed', 'fsca']

column_name_mapper = {'Latitude': 'lat', 
                      'Longitude': 'lon',
                      'vpd': 'mean_vapor_pressure_deficit',
                      'vs': 'wind_speed', 
                      'pr': 'precipitation_amount', 
                      'etr': 'potential_evapotranspiration',
                      'tmmn': 'air_temperature_tmmn',
                      'tmmx': 'air_temperature_tmmx',
                      'rmin': 'relative_humidity_rmin',
                      'rmax': 'relative_humidity_rmax',
                      'Elevation': 'Elevation',
                      'Slope': 'Slope',
                      'Aspect': 'Aspect',
                      'Curvature': 'Curvature',
                      'Northness': 'Northness',
                      'Eastness': 'Eastness',
                      'AMSR_SWE': 'SWE',
                      'AMSR_Flag': 'Flag'
}

## data on the current day
current_day_file = f'{data_dir}/testing_all_ready_{target_date}.csv_snodas_mask.csv'
current_day_data = pd.read_csv(current_day_file)
current_day_data.rename(columns=column_name_mapper, inplace=True)
current_day_data.loc[current_day_data['fsca'] > 200, 'fsca'] = np.nan
#current_day_data.loc[current_day_data['fsca'] < 0, 'fsca'] = np.nan
#current_day_data = current_day_data.dropna()
print('------ Current day:', target_date)
print('Data shape:', current_day_data.shape)
print('Data column:')
print(current_day_data.columns)

all_data = current_day_data

print('------ Reading past days...')
for i in range(1, 8):
    past_date = (target_date_dt - pd.Timedelta(days=i)).strftime('%Y-%m-%d')
    past_day_file = f'{data_dir}/testing_all_ready_{past_date}.csv_snodas_mask.csv'
    past_day_data = pd.read_csv(past_day_file)
    past_day_data.rename(columns=column_name_mapper, inplace=True)
    past_day_data = past_day_data[required_columns]
    #past_day_data.loc[past_day_data['fsca'] > 200, 'fsca'] = np.nan

    # rename columns for past days
    old_names = list(past_day_data.keys())
    new_names = [x + '_' + str(i) for x in old_names]
    past_day_data.rename(columns=dict(zip(old_names, new_names)), inplace=True)
    #past_day_data = past_day_data.dropna()

    print(past_date, past_day_data.shape)
    #all_data = all_data.append(past_day_data)
    all_data = pd.concat([all_data, past_day_data], axis=1)
    del [past_date, past_day_file, past_day_data]

print('------ Dropping NaNs...')
all_data = all_data.dropna()
nan_count = np.array(all_data.isna().sum())

print('------ Data merged!')
print('Data shape:', all_data.shape)
print('Data columns:')
print(all_data.columns)
print('Columns with NaNs:', all_data.columns[nan_count!=0])

print('------ Saving to csv...')
all_data.to_csv(f'{data_dir}/testing_all_ready_{target_date}_merged.csv_snodas_mask.csv', index=False)
print('------ Merged file saved!')