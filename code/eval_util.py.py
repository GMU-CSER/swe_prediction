
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

#### FOR SITE OBS EVALUATION ####

def stats_calculation(X, Y):
    R2 = np.corrcoef(X, Y)[0, 1]
    if R2 > 0:
        R2 = R2**2
    elif R2 < 0:
        R2 = -1 * (R2**2)   # keep the trend 
    RMSE = np.sqrt(np.mean((Y-X)**2))
    return R2, RMSE

def plot_prediction_map(
    data,
    model_option,
    date,
    pic_path
):
    coast = gpd.read_file('/groups/ESS/whung/shp_files/ne_10m_coastline/ne_10m_coastline.shp')

    lat = np.array(data['lat'])
    lon = np.array(data['lon'])
    if model_option == 'ETHole':
        swe = np.array(data['predicted_swe'])
    else:
        swe = np.array(data[f'predicted_swe_{model_option}'])
    swe[swe < 0] = 0

    fig, ax = plt.subplots(figsize=(12, 7))
    h = ax.get_position()
    ax.set_position([h.x0-0.08, h.y0, h.width+0.06, h.height])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(labelsize=24)

    plt.title(f'{date} SWE Prediction from {model_option} Model', fontsize=24, y=1.01)
    coast.plot(ax=ax, linewidth=1, color='k')

    cs = plt.scatter(lon, lat, c=swe, s=10, marker='.', cmap='turbo', vmin=0, vmax=20)

    plt.xlim([-127, -98])
    plt.xticks(np.arange(-125, -100+1, 5))
    plt.ylim([25, 50])
    plt.yticks(np.arange(25, 50+1, 5))
    plt.grid(linewidth=1, linestyle=':')

    cbax = fig.add_axes([0.75, h.y0, 0.02, h.height])
    cb   = plt.colorbar(cs, extend='max', orientation='vertical', cax=cbax)
    #cb.set_ticks()
    cb.set_label('SWE value', fontsize=24, fontweight='bold')
    cb.outline.set_linewidth(3)
    cb.ax.tick_params(labelsize=24)

    plt.savefig(f'{pic_path}/swe_prediction_{date}.png')
    plt.close()

def plot_scatter(
    X,
    Y,
    C,
    model_name,
    pic_path
):
    r2, rmse = stats_calculation(X, Y)

    tick_limit = np.ceil(np.max([X.max(), Y.max()]))

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
    cs = plt.scatter(X, Y, c=C, s=30, cmap='turbo', vmin=500, vmax=2000, alpha=0.8)
    plt.annotate(
        text='R2='+('%.4f'%r2)+', RMSE='+('%.4f'%rmse),
        xy=(0.15, 0.9),
        xycoords='figure fraction',
        fontsize=20
    )

    plt.title(f'Result Comparison ({model_name})', fontsize=24)
    plt.xlabel('Observation', fontsize=24)
    plt.ylabel(f'Model: {model_name}', fontsize=24)
    plt.xlim([0, tick_limit])
    plt.ylim([0, tick_limit])
    plt.grid(True, axis='y')

    cb = plt.colorbar(cs, extend='both')
    cb.set_label('Elevation', fontsize=24)
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
    rmse = data[f'rmse_{model_option}']

    fig, ax = plt.subplots(figsize=(10, 10))
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(labelsize=24)

    cs = plt.scatter(elv, r2, c=rmse, s=30, cmap='turbo', vmin=0, vmax=20, alpha=0.8)

    plt.title(f'Elevation-R2-RMSE ({model_option})', fontsize=24)
    plt.xlabel('Elevation', fontsize=24)
    plt.ylabel('R2', fontsize=24)
    plt.xlim([0, 4000])
    plt.ylim([-1, 1])

    cb = plt.colorbar(cs, extend='max')
    cb.set_label('RMSE', fontsize=24)
    cb.outline.set_linewidth(3)
    cb.ax.tick_params(labelsize=24)
    
    plt.tight_layout()
    plt.savefig(f'{pic_path}/elv_r2_rmse_comparison.png')
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
        plt.ylim([25, 50])
        plt.yticks(np.arange(25, 50+1, 5))
        plt.grid(linewidth=1, linestyle=':')

        cbax = fig.add_axes([0.75, h.y0, 0.02, h.height])
        cb   = plt.colorbar(cs, extend=cb_ext, orientation='vertical', cax=cbax)
        #cb.set_ticks()
        cb.set_label(item, fontsize=24, fontweight='bold')
        cb.outline.set_linewidth(3)
        cb.ax.tick_params(labelsize=24)

        plt.savefig(f'{pic_path}/site_stats_{item}.png')
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

    xx, yy = np.meshgrid(lon, lat)

    fig, ax = plt.subplots(figsize=(12, 7))
    h = ax.get_position()
    ax.set_position([h.x0-0.08, h.y0, h.width+0.06, h.height])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(labelsize=24)

    plt.title(f'{date} SWE Prediction from {prod_option} Model Product', fontsize=24, y=1.01)
    coast.plot(ax=ax, linewidth=1, color='k')

    cs = plt.pcolor(xx, yy, swe, cmap='turbo', vmin=0, vmax=20)

    plt.xlim([-127, -98])
    plt.xticks(np.arange(-125, -100+1, 5))
    plt.ylim([25, 50])
    plt.yticks(np.arange(25, 50+1, 5))
    plt.grid(linewidth=1, linestyle=':')

    cbax = fig.add_axes([0.75, h.y0, 0.02, h.height])
    cb   = plt.colorbar(cs, extend='max', orientation='vertical', cax=cbax)
    #cb.set_ticks()
    cb.set_label('SWE value', fontsize=24, fontweight='bold')
    cb.outline.set_linewidth(3)
    cb.ax.tick_params(labelsize=24)

    plt.savefig(f'{pic_path}/{prod_option}_swe_prediction_{date}.png')
    plt.close()