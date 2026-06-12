
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

#### FOR TRAINING INPUT VERIFICATION ####

def input_density_plot(
    data,
    label,
    dataset_tag,
    pic_path
):
    unit_dict = {
        'Lat': 'Degree',
        'Lon': 'Degree',
        'ELV': 'm',
        'Aspect': 'Degree',
        'Curvature': '$10^3 m^{-1}$',
        'Eastness': '-',
        'Northness': '-',
        'Slope': 'Degree',
        'TEMP': 'K',
        'VPD': 'hPa',
        'Evapo': 'm',
        'PRECIP': 'm',
        'RH': '%',
        'WS': '$m s{-1}$',
        'DOY': '-',
        'Water_year': '-',
        'Fsca': '-',
        'SWE': 'm'
    }
    xlim_dict = {
        'Lat': [25, 50],
        'Lon': [-130, -95],
        'ELV': [0, 4000],
        'Aspect': [0, 360],
        'Curvature': [-20, 20],
        'Eastness': [-1, 1],
        'Northness': [-1, 1],
        'Slope': [88, 90],
        'TEMP': [250, 310],
        'VPD': [0, 4],
        'Evapo': [0, 0.4],
        'PRECIP': [0, 0.4],
        'RH': [0, 100],
        'WS': [0, 12],
        'DOY': [1, 366],
        'Water_year': [2019-0.5, 2021+0.5],
        'Fsca': [0, 100],
        'SWE': [0, 0.4]
    }
    xtick_dict = {
        'Lat': np.arange(25, 50+1, 5),
        'Lon': np.arange(-130, -95+1, 5),
        'ELV': np.arange(0, 4000+1, 1000),
        'Aspect': np.arange(0, 360+1, 90),
        'Curvature': np.arange(-20, 20+1, 10),
        'Eastness': np.arange(-1, 1+0.1, 0.5),
        'Northness': np.arange(-1, 1+0.1, 0.5),
        'Slope': np.arange(88, 90+0.1, 0.5),
        'TEMP': np.arange(250, 310+1, 10),
        'VPD': np.arange(0, 4+1, 1),
        'Evapo': np.arange(0, 0.4+0.01, 0.1),
        'PRECIP': np.arange(0, 0.4+0.01, 0.1),
        'RH': np.arange(0, 100+1, 20),
        'WS': np.arange(0, 12+1, 2),
        'DOY': [1, 60, 120, 180, 240, 300, 366],
        'Water_year': [2019, 2020, 2021],
        'Fsca': np.arange(0, 100+1, 20),
        'SWE': np.arange(0, 0.4+0.01, 0.1)
    }

    unit = unit_dict[label]
    xlim = xlim_dict[label]
    xtick = xtick_dict[label]
    if label in ['Evapo', 'PRECIP', 'SWE']:
        data = data * 0.0254  # inch -> meter
    
    if label in ['Curvature']:
        data = data / 1000  # scaled by 1000
    
    fig, ax = plt.subplots(figsize=(12, 12))    # unit=100pixel
    #h = ax.get_position()
    #ax.set_position([h.x0+0.03, h.y0-0.01, h.width+0.01, h.height-0.01])
    ax.tick_params(labelsize=32)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)

    sns.kdeplot(data, bw_adjust=2, linewidth=4, color='orange', alpha=0.6, fill=True)

    ax.set_xlim(xlim)
    ax.set_xticks(xtick)
    ax.set_xlabel(f'{label} ({unit})', fontsize=32, fontweight='bold')
    #ax.set_ylim([0, 0.006])
    #ax.set_yticks(np.arange(0, 0.006+0.00001, 0.001))
    #ax.set_yticklabels(np.arange(0, 6+0.1, 1))
    ax.set_ylabel('Density', fontsize=32, fontweight='bold')

    plt.tight_layout()

    if dataset_tag == 'Train':
        plt.savefig(f'{pic_path}/input_density_{label}.png')
    elif dataset_tag == 'Test':
        plt.savefig(f'{pic_path}/input_density_{label}_test.png')
    plt.close()


#### FOR SITE OBS EVALUATION ####

def stats_calculation(X, Y):
    R2 = np.corrcoef(X, Y)[0, 1]
    if R2 > 0:
        R2 = R2**2
    elif R2 < 0:
        R2 = -1 * (R2**2)   # keep the trend 
    RMSE = np.sqrt(np.mean((Y-X)**2))
    return R2, RMSE

def plot_daily_boxplot(
    obs,
    pred,
    date,
    pic_path
):
    obs = [obs[i] * 0.0254 for i in range(len(obs))]  # inch -> meter
    obs_avg = np.array([np.mean(obs[i]) for i in range(len(obs))])
    obs_std = np.array([np.std(obs[i]) for i in range(len(obs))])

    xtick = np.arange(len(date))
    xlim = [xtick[0] - 0.5, xtick[-1] + 0.5]
    interval = [-0.3, -0.1, 0.1, 0.3]
    labels = ['MSELoss', 'SWELoss', 'RMSELoss', 'ET']
    colors = ['pink', 'crimson', 'orange', 'royalblue']

    fig, ax = plt.subplots(figsize=(15, 9))
    h = ax.get_position()
    ax.set_position([h.x0-0.04, h.y0+0.011, h.width+0.1, h.height])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(labelsize=24)

    plt.title(f'Daily Average SWE Prediction', fontsize=24, y=1.01)

    for i in range(len(pred)):
        data = pred[i]
        data = [data[j][~np.isnan(data[j])] * 0.0254 for j in range(len(data))]  # inch -> meter
        plt.boxplot(
            data,
            positions=xtick + interval[i],
            widths=0.2,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(linewidth=2, facecolor=colors[i]),
            capprops=dict(linewidth=2),
            medianprops=dict(linewidth=2, color='k'),
            label=labels[i]
        )
        del data

    plt.plot(xtick, obs_avg, linewidth=3, color='grey')
    plt.errorbar(xtick, obs_avg, yerr=obs_std, linewidth=3, color='grey', fmt='o', label='Observation')

    ax.set_xlim(xlim)
    ax.set_xticks(xtick)
    ax.set_xticklabels(date, rotation=90)
    ax.set_ylim([-0.05, 1])
    ax.set_yticks(np.arange(0, 1+0.1, 0.2))
    ax.set_ylabel('SWE (m)', fontsize=24, fontweight='bold')
    plt.grid(axis='y', linewidth=1, linestyle=':')

    plt.legend(loc='upper left', ncol=3, prop={'size':24})
    #plt.tight_layout()
    plt.savefig(f'{pic_path}/daily_boxplot.png')
    plt.close()

def plot_prediction_map(
    data,
    model_option,
    date,
    pic_path
):
    coast = gpd.read_file('/groups/ESS/whung/shp_files/ne_10m_coastline/ne_10m_coastline.shp')

    lat = np.array(data['lat'])
    lon = np.array(data['lon'])
    if model_option == 'ExtraTree':
        swe = np.array(data['predicted_swe'])
    else:
        swe = np.array(data[f'predicted_swe_{model_option}'])
    swe[swe < 0] = 0

    swe = swe * 0.0254  # inch -> meter
    #print(swe.min(), swe.max())

    fig, ax = plt.subplots(figsize=(12, 7))
    h = ax.get_position()
    ax.set_position([h.x0-0.08, h.y0, h.width+0.06, h.height])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(labelsize=24)

    plt.title(f'{date} SWE Prediction from {model_option} Model', fontsize=24, y=1.01)
    coast.plot(ax=ax, linewidth=1, color='k')

    cs = plt.scatter(lon, lat, c=swe, s=10, marker='.', cmap='turbo', vmin=0, vmax=1)

    plt.xlim([-127, -98])
    plt.xticks(np.arange(-125, -100+1, 5))
    plt.ylim([30, 50])
    plt.yticks(np.arange(30, 50+1, 5))
    plt.grid(linewidth=1, linestyle=':')

    cbax = fig.add_axes([0.82, h.y0, 0.02, h.height])
    cb   = plt.colorbar(cs, extend='max', orientation='vertical', cax=cbax)
    #cb.set_ticks()
    cb.set_label('SWE value (m)', fontsize=24, fontweight='bold')
    cb.outline.set_linewidth(3)
    cb.ax.tick_params(labelsize=24)

    plt.savefig(f'{pic_path}/swe_prediction_{date}.png')
    plt.close()

def plot_scatter(
    X,
    Y,
    model_name,
    pic_path
):
    X = np.array(X) * 0.0254  # in -> m
    Y = np.array(Y) * 0.0254  # in -> m
    r2, rmse = stats_calculation(X, Y)

    # data density
    xy = np.vstack([X, Y])
    Z = gaussian_kde(xy)(xy)
    idx = Z.argsort()
    x, y, z = X[idx], Y[idx], Z[idx]

    #tick_limit = np.ceil(np.max([X.max(), Y.max()]))
    tick_limit = 1

    fig, ax = plt.subplots(figsize=(10, 10))
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(labelsize=24)

    plt.plot(np.arange(0, tick_limit+0.1, 0.1),
        np.arange(0, tick_limit+0.1, 0.1),
        linewidth=1,
        linestyle=':',
        color='k'
    )
    cs = plt.scatter(x, y, c=z, s=30, cmap='turbo', alpha=0.8)
    plt.annotate(
        text='R2='+('%.4f'%r2)+', RMSE='+('%.4f'%rmse),
        xy=(0.15, 0.9),
        xycoords='figure fraction',
        fontsize=20
    )

    plt.title(f'Result Comparison ({model_name})', fontsize=24)
    plt.xlabel('Observation (m)', fontsize=24)
    plt.ylabel(f'Model: {model_name} (m)', fontsize=24)
    plt.xlim([0, tick_limit])
    plt.ylim([0, tick_limit])
    plt.grid(True, axis='y')

    cb = plt.colorbar(cs)
    cb.set_label('Density', fontsize=24)
    cb.outline.set_linewidth(3)
    cb.ax.tick_params(labelsize=24)

    plt.tight_layout()
    plt.savefig(f'{pic_path}/obs_site_comparison.png')
    plt.close()

def plot_elv_r2_rmse(
    data,
    model_option,
    pic_path
):
    elv = data['site_elv']
    r2 = data[f'r2_{model_option}']
    rmse = data[f'rmse_{model_option}'] * 0.0254  # in -> m

    fig, ax = plt.subplots(figsize=(10, 10))
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(labelsize=24)

    cs = plt.scatter(elv, r2, c=rmse, s=30, cmap='turbo', vmin=0, vmax=1, alpha=0.8)

    plt.title(f'Elevation-R2-RMSE ({model_option})', fontsize=24)
    plt.xlabel('Elevation', fontsize=24)
    plt.ylabel('R2', fontsize=24)
    plt.xlim([0, 4000])
    plt.ylim([-1, 1])

    cb = plt.colorbar(cs, extend='max')
    cb.set_label('RMSE (m)', fontsize=24)
    cb.outline.set_linewidth(3)
    cb.ax.tick_params(labelsize=24)
    
    plt.tight_layout()
    plt.savefig(f'{pic_path}/elv_r2_rmse_comparison.png')
    plt.close()

def plot_rmse_r2_swe(
    data,
    model_option,
    pic_path
):
    r2 = data[f'r2_{model_option}']
    rmse = data[f'rmse_{model_option}'] * 0.0254  # in -> m
    swe = data['site_swe'] * 0.0254  # in -> m

    fig, ax = plt.subplots(figsize=(10, 10))
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(labelsize=24)

    cs = plt.scatter(rmse, r2, c=swe, s=30, cmap='turbo', vmin=0, vmax=1, alpha=0.8)

    plt.title(f'RMSE-R2-SWE ({model_option})', fontsize=24)
    plt.xlabel('RMSE', fontsize=24)
    plt.ylabel('R2', fontsize=24)
    plt.xlim([0, 1])
    plt.ylim([-1, 1])

    cb = plt.colorbar(cs, extend='max')
    cb.set_label('Average SWE (m)', fontsize=24)
    cb.outline.set_linewidth(3)
    cb.ax.tick_params(labelsize=24)
    
    plt.tight_layout()
    plt.savefig(f'{pic_path}/rmse_r2_swe_comparison.png')
    plt.close()

def plot_stats_map(
    data,
    model_option,
    pic_path
):
    coast = gpd.read_file('/groups/ESS/whung/shp_files/ne_10m_coastline/ne_10m_coastline.shp')
    
    lat = np.array(data['site_lat'])
    lon = np.array(data['site_lon'])
    stats = {
        'R2': np.array(data[f'r2_{model_option}']),
        'RMSE': np.array(data[f'rmse_{model_option}']) * 0.0254  # in -> m
    }

    for item in stats:
        if item == 'R2':
            vlim = [-1, 1]
            cmap = 'seismic'
            cb_ext = 'neither'
        elif item == 'RMSE':
            vlim = [0, 1]
            cmap = 'turbo'
            cb_ext = 'max'

        fig, ax = plt.subplots(figsize=(12, 7))
        h = ax.get_position()
        ax.set_position([h.x0-0.08, h.y0, h.width+0.06, h.height])
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
        ax.tick_params(labelsize=24)

        plt.title(f'{item} of SWE Prediction from {model_option} Model', fontsize=24)
        coast.plot(ax=ax, linewidth=1, color='k')

        cs = plt.scatter(lon, lat, c=stats[item], s=100, marker='.', cmap=cmap, vmin=vlim[0], vmax=vlim[1])

        plt.xlim([-127, -98])
        plt.xticks(np.arange(-125, -100+1, 5))
        plt.ylim([30, 50])
        plt.yticks(np.arange(30, 50+1, 5))
        plt.grid(linewidth=1, linestyle=':')

        cbax = fig.add_axes([0.82, h.y0, 0.02, h.height])
        cb   = plt.colorbar(cs, extend=cb_ext, orientation='vertical', cax=cbax)
        #cb.set_ticks()
        cb.set_label(item, fontsize=24, fontweight='bold')
        cb.outline.set_linewidth(3)
        cb.ax.tick_params(labelsize=24)

        plt.savefig(f'{pic_path}/site_stats_{item}.png')
        plt.close()

def plot_avg_swe_map(
    data,
    pic_path
):
    coast = gpd.read_file('/groups/ESS/whung/shp_files/ne_10m_coastline/ne_10m_coastline.shp')
    
    lat = np.array(data['site_lat'])
    lon = np.array(data['site_lon'])
    swe = np.array(data['site_swe']) * 0.0254  # in -> m

    fig, ax = plt.subplots(figsize=(12, 7))
    h = ax.get_position()
    ax.set_position([h.x0-0.08, h.y0, h.width+0.06, h.height])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(labelsize=24)

    plt.title(f'Average SWE at SNOTEL sites in Jan-Feb 2025', fontsize=24)
    coast.plot(ax=ax, linewidth=1, color='k')

    cs = plt.scatter(lon, lat, c=swe, s=100, marker='.', cmap='turbo', vmin=0, vmax=1)

    plt.xlim([-127, -98])
    plt.xticks(np.arange(-125, -100+1, 5))
    plt.ylim([30, 50])
    plt.yticks(np.arange(30, 50+1, 5))
    plt.grid(linewidth=1, linestyle=':')

    cbax = fig.add_axes([0.82, h.y0, 0.02, h.height])
    cb   = plt.colorbar(cs, extend='max', orientation='vertical', cax=cbax)
    #cb.set_ticks()
    cb.set_label('SWE (m)', fontsize=24, fontweight='bold')
    cb.outline.set_linewidth(3)
    cb.ax.tick_params(labelsize=24)

    plt.savefig(f'{pic_path}/site_avg_swe.png')
    plt.close()

#### FOR MODEL PRODUCTS EVALUATION ####

def plot_model_products(
    data,
    prod_option,
    date,
    pic_path
):
    coast = gpd.read_file('/groups/ESS/whung/shp_files/ne_10m_coastline/ne_10m_coastline.shp')

    lat = data['lat']
    lon = data['lon']
    swe = data['swe']
    swe[swe < 0] = 0
    swe = swe * 0.0254  # in -> m

    xx, yy = np.meshgrid(lon, lat)

    fig, ax = plt.subplots(figsize=(12, 7))
    h = ax.get_position()
    ax.set_position([h.x0-0.08, h.y0, h.width+0.06, h.height])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(labelsize=24)

    plt.title(f'{date} SWE Prediction from {prod_option} Model Product', fontsize=24, y=1.01)
    coast.plot(ax=ax, linewidth=1, color='k')

    cs = plt.pcolor(xx, yy, swe, cmap='turbo', vmin=0, vmax=1)

    plt.xlim([-127, -98])
    plt.xticks(np.arange(-125, -100+1, 5))
    plt.ylim([25, 50])
    plt.yticks(np.arange(25, 50+1, 5))
    plt.grid(linewidth=1, linestyle=':')

    cbax = fig.add_axes([0.75, h.y0, 0.02, h.height])
    cb   = plt.colorbar(cs, extend='max', orientation='vertical', cax=cbax)
    #cb.set_ticks()
    cb.set_label('SWE value (m)', fontsize=24, fontweight='bold')
    cb.outline.set_linewidth(3)
    cb.ax.tick_params(labelsize=24)

    plt.savefig(f'{pic_path}/{prod_option}_swe_prediction_{date}.png')
    plt.close()

def plot_model_scatter(
    X,
    Y,
    prod_option,
    pic_path
):
    X = np.array(X) * 0.0254  # in -> m
    Y = np.array(Y) * 0.0254  # in -> m
    r2, rmse = stats_calculation(X, Y)

    # data density
    xy = np.vstack([X, Y])
    Z = gaussian_kde(xy)(xy)
    idx = Z.argsort()
    x, y, z = X[idx], Y[idx], Z[idx]

    #tick_limit = np.ceil(np.max([X.max(), Y.max()]))
    tick_limit = 1

    fig, ax = plt.subplots(figsize=(10, 10))
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(labelsize=24)

    plt.plot(np.arange(tick_limit),
        np.arange(tick_limit),
        linewidth=1,
        linestyle=':',
        color='k'
    )
    cs = plt.scatter(x, y, c=z, s=30, cmap='turbo', alpha=0.8)
    plt.annotate(
        text='R2='+('%.4f'%r2)+', RMSE='+('%.4f'%rmse),
        xy=(0.15, 0.9),
        xycoords='figure fraction',
        fontsize=20
    )

    plt.title(f'Site Observation Comparison ({prod_option})', fontsize=24)
    plt.xlabel('Observation (m)', fontsize=24)
    plt.ylabel(f'{prod_option} Model (m)', fontsize=24)
    plt.xlim([0, tick_limit])
    plt.ylim([0, tick_limit])
    plt.grid(True, axis='y')

    cb = plt.colorbar(cs)
    cb.set_label('Density', fontsize=24)
    cb.outline.set_linewidth(3)
    cb.ax.tick_params(labelsize=24)
    
    plt.tight_layout()
    plt.savefig(f'{pic_path}/{prod_option}_site_comparison.png')
    plt.close()

 