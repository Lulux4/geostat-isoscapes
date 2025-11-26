import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt 
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import xarray as xr
import numpy as np
from . import utils, geostat_utils as gutils

def plot_isoscape_latlon_platecarree(dataarray_slice: xr.DataArray,
                                     time: str, 
                                     title :str = 'Values of d18Op', 
                                     countries_borders : bool =False,
                                     cbar_label :str = r'$\delta^{18}$O [‰]',
                                     )-> tuple[Figure,Axes]:
    """ Plot the given isotope data array slice in platecarree projection.
    """
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()})
    
    dataarray_slice.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        cbar_kwargs={"label": cbar_label},
        robust=True # robust to large outliers
    ) #type: ignore
    
    if countries_borders :
        country_borders = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_boundary_lines_land',
        scale='50m',
        facecolor='none')
        ax.add_feature(country_borders, edgecolor='gray') #type:ignore

    ax.coastlines() # type: ignore
    valid = dataarray_slice.where(~np.isnan(dataarray_slice), drop=True)
    ax.set_extent([             # type: ignore
        float(valid.lon.min()),
        float(valid.lon.max()),
        float(valid.lat.min()),
        float(valid.lat.max())
    ], crs=ccrs.PlateCarree()) 
    # title with the year and the month
    ax.set_title(f"{title} - {time}")
    return fig, ax

def plot_isoscape_latlon_platecarree_df(
        df,
        time: str,
        title: str = "Values of d18O",
        countries_borders: bool = False
    ):
    """
    Plot the given dataframe with columns: lat, lon, d18O on PlateCarree projection.
    """
    fig, ax = plt.subplots(figsize=(10, 5),
                           subplot_kw={"projection": ccrs.PlateCarree()})

    # Scatter plot of irregular points
    sc = ax.scatter(
        df["lon"],
        df["lat"],
        c=df["d18O"],
        cmap="viridis",
        transform=ccrs.PlateCarree(),
        s=20,
        edgecolor="none"
    )

    cbar = plt.colorbar(sc, ax=ax, label="d18O")
    if countries_borders:
        country_borders = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_0_boundary_lines_land',
            scale='50m',
            facecolor='none'
        )
        ax.add_feature(country_borders, edgecolor='gray') # type:ignore
    ax.coastlines() # type:ignore
    valid = df.dropna(subset=["lat", "lon", "d18O"])
    ax.set_extent([         #type:ignore
        valid["lon"].min(),
        valid["lon"].max(),
        valid["lat"].min(),
        valid["lat"].max()
    ], crs=ccrs.PlateCarree())
    ax.set_title(f"{title} - {time}")

    return fig, ax


def plot_projected_data(coords_proj,vals_s, projection_str='Mercator',robust=True):
    if projection_str == 'Mercator':
        projection = ccrs.Mercator()
    
    # nan mask
    nan_mask = np.isnan(vals_s)
    vals_s= vals_s[~nan_mask]
    coords_proj = coords_proj[~nan_mask,:]
    if robust:
        # clip outliers to make robust colormap
        q_low, q_high = np.quantile(vals_s, [0.005, 0.995])
        if q_low == q_high:
            vmin, vmax = float(vals_s.min()), float(vals_s.max())
        else:
            vmin, vmax = float(q_low), float(q_high)
        vals_s = np.clip(vals_s, vmin, vmax)

    fig, ax = plt.subplots(subplot_kw={"projection": projection})
    sc = ax.scatter(coords_proj[:,0],coords_proj[:,1], c=vals_s, s=20, cmap="viridis", transform=projection, alpha=0.9, edgecolor="k", linewidth=0.1)
    ax.coastlines(resolution="50m") # type: ignore
    ax.set_title(f"d18Op values ({projection_str})")
    plt.colorbar(sc, ax=ax, label="d18Op")
        
    return fig, ax

def plot_variogram_from_bins_and_gamma(centers,
                                       gamma,
                                       time : str, 
                                       counts = None, 
                                       std_counts=None,
                                       min_pairs=30, 
                                       plot_model : bool = True, 
                                       model_name : str|None = None, 
                                       model_fct = None,
                                       model_params = None,
                                       figsize: tuple[int,int]=(10,5),
                                       ax_ = None,
                                       save_name : str | None = None
                                       ) -> tuple[Figure,Axes] | Axes :
    """ Plot an empirical variogram from given bin centers and semivariances values.
    Overlays pairs number per bin if counts is given. Does not plot bins with less than min_pairs if counts is given.
    """
    if counts is not None :
        reliable = counts >= min_pairs
        centers = centers[reliable]
        gamma = gamma[reliable]
        if counts is not None:
            counts = counts[reliable]
    if ax_ is None :
        fig, ax = plt.subplots(figsize=figsize)
    else :
        ax = ax_
    
    ax.plot(centers, gamma, 'o--', color='C0',linewidth=1,markersize=4)
    

    # Overlay model if wanted and given
    if (plot_model) and (model_name is not None) and (model_fct is not None):
        h = np.arange(0,centers.max(),10000)
        ax.plot(h,model_fct(h),'-', color='C1', linewidth=1, label=f'{model_name} fit')
        sill,range_,nugget= None,None,None
        
        if (model_name != 'composite') and (model_params is not None) :
            range_ = model_params['range']
            sill = model_params['sill']+model_params['nugget']
            nugget = model_params['nugget']
            ax.vlines(range_,0,max(gamma),color="#FF1E00",linestyle='--',alpha=0.2,label=f"range")
            ax.hlines(sill,0,max(centers),color="#BD6D12",linestyle='--',alpha=0.2,label=f"sill")
        
        elif (model_name =='composite') and (model_params is not None) :
            range_ = gutils.effective_range(centers,model_fct,0.95)
            sill = model_params['sill1']+model_params['sill2']+model_params['nugget']
            nugget= model_params['nugget']
            ax.vlines(range_,0,max(gamma),color='#FF1E00',linestyle='--',alpha=0.2,label=f'effective range')
            ax.hlines(sill,0,max(centers),color='#BD6D12',linestyle='--',alpha=0.2,label=f"total sill")
        if (range_ is not None) & (sill is not None) & (nugget is not None):
            weights = None
            if counts is not None : 
                weights = gutils.get_weights_from_pair_counts(counts)
            r2 = utils.compute_r2(gamma, model_fct(centers), weights=weights)
            textstr = f"Range: {range_:.1e} m\nSill: {sill:.2f} ‰\nNugget: {nugget:.2f} ‰\nR²: {r2:.2f}"
            ax.text(0.835, 0.22, textstr,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='bottom',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8)
                    )
        if ax_ is None : plt.legend(loc='lower right')
    
    width = centers[1]-centers[0]
    # Overlay bins counts if given
    if counts is not None:
        # Overlay pair counts
        ax2 = ax.twinx()
        ax2.bar(centers, counts, width = width,
                color='red', alpha=0.1, label='Pair counts')
        ax2.set_ylabel('Number of pairs', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        if std_counts is not None :
            ax2.bar(centers, std_counts, width = width,
                    color='blue', alpha=0.1, label='Pair counts std')
    ax.set_xlim(0, centers.max()+width)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.4)
    ax.set_xlabel("Lag distance [m]")
    ax.set_ylabel("Semivariance")
    
    if ax_ is None :
        plt.title(f"Empirical variogram - {time}")
        if save_name is not None:
            plt.savefig(save_name,dpi=500,bbox_inches='tight')
        return fig,ax
    return ax

def plot_elevation_map( df, countries_borders: bool = False) -> tuple[Figure, Axes]:
    """ Plot an elevation map from given latitudes, longitudes and elevation grid.
    """
    fig, ax = plt.subplots(figsize=(10, 5),
                           subplot_kw={"projection": ccrs.PlateCarree()})

    sc = ax.scatter(df['lon'], df['lat'], c=df['elevation'], cmap='terrain', s=2, edgecolor=None)
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', label='Elevation (m)')
    if countries_borders:
        country_borders = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_0_boundary_lines_land',
            scale='50m',
            facecolor='none'
        )
        ax.add_feature(country_borders, edgecolor='gray') #type:ignore
    ax.coastlines() # type:ignore
    return fig, ax

def plot_dist_to_coast_map(df,countries_borders: bool = False) -> tuple[Figure, Axes]:
    """ Plot a distance to coast map from given df containing latitudes, longitudes and distance to coast grid.
    """
    fig, ax = plt.subplots(figsize=(10, 5),
                           subplot_kw={"projection": ccrs.PlateCarree()})

    sc = ax.scatter(df['lon'], df['lat'], c=df['dist_coast_m'], cmap='viridis', s=2, edgecolor=None)
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', label='Distance to coast (km)')
    if countries_borders:
        country_borders = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_0_boundary_lines_land',
            scale='50m',
            facecolor='none'
        )
        ax.add_feature(country_borders, edgecolor='gray') #type:ignore
    ax.coastlines() # type:ignore
    return fig, ax