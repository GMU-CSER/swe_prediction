import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = '/groups/ESS/whung/swe_gnn/data'
train_file = f'{data_path}/all_points_final_merged_training_snodas_mask_resnet_all_batch.csv'

## reading train data
print('\n---- Reading training variables...')
train_data = pd.read_csv(train_file, usecols=['station_name', 'Elevation', 'air_temperature_tmmx', 'air_temperature_tmmn', 'precipitation_amount', 'mean_vapor_pressure_deficit', 'wind_speed', 'swe_value'])

train_data.loc[train_data['air_temperature_tmmx'] < 250, 'air_temperature_tmmx'] = np.nan
train_data.loc[train_data['air_temperature_tmmn'] < 250, 'air_temperature_tmmn'] = np.nan
train_data.loc[train_data['Elevation'] < 0, 'Elevation'] = np.nan
train_data.loc[train_data['swe_value'] < 0, 'swe_value'] = np.nan
train_data = train_data.dropna()

train_data.loc[train_data['station_name'] == '0', 'station_name'] = 0
train_data = train_data.loc[train_data.station_name != 0]  # remove fake sites

train_data['temp_avg'] = (train_data['air_temperature_tmmx'] + train_data['air_temperature_tmmn'])/2
train_data = train_data.groupby(['station_name']).agg(
    temp=('temp_avg', 'mean'),
    precip=('precipitation_amount', 'mean'),
    elv=('Elevation', 'mean'),
    vpd=('mean_vapor_pressure_deficit', 'mean'),
    ws=('wind_speed', 'mean'),
    swe=('swe_value', 'mean')
).reset_index()
train_data['precip'] = train_data['precip'] * 0.0254  # in -> m
train_data['swe'] = train_data['swe'] * 0.0254  # in -> m
print('---- Train sites:', train_data.shape)
print(train_data)

elv = train_data['elv']
temp = train_data['temp']
precip = train_data['precip']
swe = train_data['swe']

idx = swe > 0
elv = elv[idx]
temp = temp[idx]
precip = precip[idx]
swe = swe[idx]

print('Elv:', elv.min(), elv.max())
print('Temp:', temp.min(), temp.max())
print('Precip:', precip.min(), precip.max())
print('SWE:', swe.min(), swe.max())


## elv-grouped boxplots
print('\n---- Plotting elv-grouped boxplots...')
elv_bound = [0, 1000, 2000, 3000, 4000]
temp_group = [[]]*(len(elv_bound)-1)
precip_group = [[]]*(len(elv_bound)-1)
swe_group = [[]]*(len(elv_bound)-1)

for i in range(len(elv_bound)-1):
    idx = (elv >= elv_bound[i]) & (elv < elv_bound[i+1])
    temp_group[i] = temp[idx]
    precip_group[i] = precip[idx]
    swe_group[i] = swe[idx]
    del idx

plot_tick = np.arange(len(elv_bound)-1) + 0.5
xtick = np.arange(len(elv_bound))
xlim = [xtick[0], xtick[-1]]
xticklabel = elv_bound
groups = [swe_group, temp_group, precip_group]
intervals = [-0.2, 0, 0.2]
facecolors = ['lightgrey', 'skyblue', 'gold']
edgecolors = ['k', 'b', 'orange']
labels = ['SWE', 'TEMP', 'PRECIP']

fig, ax1 = plt.subplots(figsize=(12, 10))
ax2 = ax1.twinx()  
h = ax1.get_position()
ax1.set_position([h.x0-0.02, h.y0, h.width, h.height+0.08])
ax2.set_position([h.x0-0.02, h.y0, h.width, h.height+0.08])
for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(3)
    ax2.spines[axis].set_linewidth(3)
ax1.tick_params(labelsize=28)
ax2.tick_params(labelsize=28)

handles = [[]]*len(groups)
for i in range(len(groups)):
    if labels[i] == 'TEMP':
        h = ax2.boxplot(
            groups[i],
            positions=plot_tick + intervals[i],
            widths=0.18,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(linewidth=2, facecolor=facecolors[i], edgecolor=edgecolors[i], alpha=0.8),
            whiskerprops=dict(linewidth=2, color=edgecolors[i]),
            capprops=dict(linewidth=2, color=edgecolors[i]),
            medianprops=dict(linewidth=2, color=edgecolors[i]),
            label=labels[i]
        )
    else:
        h = ax1.boxplot(
            groups[i],
            positions=plot_tick + intervals[i],
            widths=0.18,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(linewidth=2, facecolor=facecolors[i], edgecolor=edgecolors[i], alpha=0.8),
            whiskerprops=dict(linewidth=2, color=edgecolors[i]),
            capprops=dict(linewidth=2, color=edgecolors[i]),
            medianprops=dict(linewidth=2, color=edgecolors[i]),
            label=labels[i]
        )
    
    handles[i] = h["boxes"][0]

ax1.set_xlim(xlim)
ax1.set_xticks(xtick)
ax1.set_xticklabels(xticklabel)
ax1.set_xlabel('Elevation (m)', fontsize=28)
#ax1.set_ylim([-0.05, 1])
#ax1.set_yticks(np.arange(0, 1+0.1, 0.2))
ax1.set_ylabel('SWE/PRECIP (m)', fontsize=28)

ax2.set_xlim(xlim)
ax2.set_xticks(xtick)
#ax2.set_ylim([-0.05, 1])
#ax2.set_yticks(np.arange(0, 1+0.1, 0.2))
ax2.set_ylabel('TEMP (K)', fontsize=32)

plt.legend(handles, labels, loc='upper right', prop={'size':24})
#plt.tight_layout()
plt.savefig(f'{data_path}/boxplot_elv_temp_precip.png')
plt.close()

del [elv_bound, swe_group, temp_group, precip_group]
del [plot_tick, xtick, xlim, xticklabel, groups, intervals, facecolors, edgecolors, labels]


## elv comparison
print('\n---- Plotting elv-based scatter plots...')
xlim = [0, 4000]
xtick = np.arange(0, 4000+1, 1000)

data = [temp, precip]
label = ['TEMP (K)', 'PRECIP (m)']
fname = ['temp', 'precip']
ylim = [[274, 284], [0, 0.3]]
ytick = [np.arange(274, 284+1, 2), np.arange(0, 0.3+0.01, 0.1)]

for i in range(len(data)):
    r = np.corrcoef(elv, data[i])[0, 1]
    a, b = np.polyfit(elv, data[i], 1)

    fig, ax = plt.subplots(figsize=(10, 10))
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(labelsize=28)

    plt.scatter(elv, data[i], marker='o', c='orange', edgecolor='darkorange', s=30, alpha=0.8)
    plt.plot(xtick, a*xtick+b, linewidth=2, color='grey')
    plt.annotate(
        text='R='+('%.4f'%r),
        xy=(0.2, 0.85),
        xycoords='figure fraction',
        fontsize=28
    )

    plt.xlabel('Elevation (m)', fontsize=28)
    plt.ylabel(label[i], fontsize=28)
    plt.xlim(xlim)
    plt.ylim(ylim[i])
    plt.yticks(ytick[i])

    plt.tight_layout()
    plt.savefig(f'{data_path}/elv_{fname[i]}_comparison.png')
    plt.close()
del [xlim, xtick, ylim, ytick, data, label, fname]


## swe comparison
print('\n---- Plotting swe-based scatter plots...')
ylim = [0, 0.8]
ytick = np.arange(0, 0.8+0.1, 0.2)

data = [elv, temp, precip]
label = ['ELV (m)', 'TEMP (K)', 'PRECIP (m)']
fname = ['elv' ,'temp', 'precip']
xlim = [[0, 4000], [274, 284], [0, 0.3]]
xtick = [np.arange(0, 4000+1, 1000), np.arange(274, 284+1, 2), np.arange(0, 0.3+0.05, 0.1)]

for i in range(len(data)):
    r = np.corrcoef(data[i], swe)[0, 1]
    a, b = np.polyfit(data[i], swe, 1)

    fig, ax = plt.subplots(figsize=(10, 10))
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(labelsize=28)

    plt.scatter(data[i], swe, marker='o', c='royalblue', edgecolor='b', s=30, alpha=0.8)
    plt.plot(xtick[i], a*xtick[i]+b, linewidth=2, color='grey')
    plt.annotate(
        text='R='+('%.4f'%r),
        xy=(0.2, 0.85),
        xycoords='figure fraction',
        fontsize=28
    )

    plt.xlabel(label[i], fontsize=28)
    plt.ylabel('SWE (m)', fontsize=28)
    plt.xlim(xlim[i])
    plt.ylim(ylim)
    plt.xticks(xtick[i])

    plt.tight_layout()
    plt.savefig(f'{data_path}/swe_{fname[i]}_comparison.png')
    plt.close()
del [xlim, xtick, ylim, ytick, data, label, fname]


## SNOTEL-GHCN comparison
print('\n---- Plotting SNOTEL-GHCN distributions...')
import seaborn as sns

snotel = []
ghcn = []
for item in train_data['station_name']:
    if ('US' in item) or ('CA' in item):
        ghcn.append(item)
    else:
        snotel.append(item)
snotel_data = train_data[train_data['station_name'].isin(snotel)]
ghcn_data = train_data[train_data['station_name'].isin(ghcn)]
print('snotel', snotel_data.shape)
print('ghcn', ghcn_data.shape)

var = ['elv' ,'temp', 'precip', 'vpd', 'ws', 'swe']
label = ['ELV (m)', 'TEMP (K)', 'PRECIP (m)', 'VPD (hPa)', 'WS ($m^{-1}$)', 'SWE (m)']
xlim = [[0, 4000], [274, 284], [0, 0.3], [0, 4], [0, 12], [0, 0.8]]
xtick = [np.arange(0, 4000+1, 1000), np.arange(274, 284+1, 2), np.arange(0, 0.3+0.05, 0.1), np.arange(0, 4+0.1, 0.5), np.arange(0, 12+1, 2), np.arange(0, 0.8+0.1, 0.2)]

for i in [5]:  #range(len(var)):
    fig, ax = plt.subplots(figsize=(12, 12))    # unit=100pixel
    #h = ax.get_position()
    #ax.set_position([h.x0+0.03, h.y0-0.01, h.width+0.01, h.height-0.01])
    ax.tick_params(labelsize=32)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)

    #print(snotel_data[var[i]])
    print(np.unique(ghcn_data[var[i]]))
    sns.kdeplot(snotel_data[var[i]], bw_adjust=2, color='orange', alpha=0.6, fill=True, label='SNOTEL')
    sns.kdeplot(ghcn_data[var[i]], bw_adjust=2, linewidth=4, color='royalblue', label='GHCN')

    ax.set_xlim(xlim[i])
    ax.set_xticks(xtick[i])
    ax.set_xlabel(label[i], fontsize=28)
    #ax.set_ylim([0, 0.006])
    #ax.set_yticks(np.arange(0, 0.006+0.00001, 0.001))
    #ax.set_yticklabels(np.arange(0, 6+0.1, 1))
    ax.set_ylabel('Density', fontsize=28)

    plt.legend(loc='upper left', prop={'size':24})
    plt.tight_layout()

    plt.tight_layout()
    plt.savefig(f'{data_path}/snotel_ghcn_comparison_{var[i]}.png')
    plt.close()
