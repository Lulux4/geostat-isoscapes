import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt 
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import xarray as xr
import numpy as np

def plot_isoscape_latlon_platecarree(dataarray_slice: xr.DataArray,
                                     time: str, 
                                     title :str = 'Values of d18Op', 
                                     countries_borders : bool =False
                                     )-> tuple[Figure,Axes]:
    ''' Plot the given isotope data array slice in platecarree projection.
    '''
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()})
    
    dataarray_slice.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        cbar_kwargs={"label": dataarray_slice.name},
        robust=True
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

def plot_projected_data(coords_proj,vals_s, projection_str='Mercator'):
    if projection_str == 'Mercator':
        projection = ccrs.Mercator()
    
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
                                       figsize: tuple[int,int]=(10,5)
                                       ) -> tuple[Figure,Axes]:
    ''' Plot an empirical variogram from given bin centers and semivariances values.
    Overlays pairs number per bin if counts is given. Does not plot bins with less than min_pairs if counts is given.
    '''
    if counts is not None :
        reliable = counts >= min_pairs
        centers = centers[reliable]
        gamma = gamma[reliable]
        if counts is not None:
            counts = counts[reliable]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(centers/1000, gamma, 'o--', color='C0',linewidth=1,markersize=4)
    ax.set_xlabel("Lag distance (km)")
    ax.set_ylabel("Semivariance")

    # Overlay model if wanted and given
    if (plot_model) and (model_name is not None) and (model_fct is not None):
        ax.plot(centers/1000,model_fct(centers),'-', color='C1', linewidth=1, label=f'{model_name} fit')
        plt.legend()
    
    width = (centers[1]-centers[0])/1000
    # Overlay bins counts if given
    if counts is not None:
        # Overlay pair counts
        ax2 = ax.twinx()
        ax2.bar(centers/1000, counts, width = width,
                color='red', alpha=0.1, label='Pair counts')
        ax2.set_ylabel('Number of pairs', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        if std_counts is not None :
            ax2.bar(centers/1000, std_counts, width = width,
                    color='blue', alpha=0.1, label='Pair counts std')
    
    plt.title(f"Empirical variogram — {time}")
    plt.xlim(0, centers.max()/1000+width)
    plt.grid(True, alpha=0.4)
    plt.show()

    return fig,ax