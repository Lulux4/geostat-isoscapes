import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt 
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import xarray as xr
import numpy as np
from . import utils, geostat_utils as gutils
import pandas as pd
import plotly.graph_objects as go

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
        time: str = '',
        title: str = "Values of d18O",
        countries_borders: bool = False,
        lat_col :str = 'lat',
        lon_col :str = 'lon',
        qty_col :str = 'd18Op'
    ):
    """
    Plot the given dataframe with columns: lat, lon, d18O on PlateCarree projection.
    """
    fig, ax = plt.subplots(figsize=(10, 5),
                           subplot_kw={"projection": ccrs.PlateCarree()})

    # Scatter plot of irregular points
    sc = ax.scatter(
        df[lon_col],
        df[lat_col],
        c=df[qty_col],
        cmap="viridis",
        transform=ccrs.PlateCarree(),
        s=20,
        edgecolor="none"
    )

    cbar = plt.colorbar(sc, ax=ax, label=qty_col)
    if countries_borders:
        country_borders = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_0_boundary_lines_land',
            scale='50m',
            facecolor='none'
        )
        ax.add_feature(country_borders, edgecolor='gray') # type:ignore
    ax.coastlines() # type:ignore
    valid = df.dropna(subset=[lat_col, lon_col, qty_col])
    ax.set_extent([         #type:ignore
        valid[lon_col].min(),
        valid[lon_col].max(),
        valid[lat_col].min(),
        valid[lat_col].max()
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
                                       textbox_loc = (0.835, 0.22),
                                       legend_loc='lower right',
                                       ax_ = None,
                                       save_name : str | None = None,
                                       verbose : bool =  False
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
        if verbose : print('plotting model')
        h = np.arange(0,centers.max(),10000)
        legend_model = model_name 
        ax.plot(h,model_fct(h),'-', color='C1', linewidth=1, label=f'{legend_model} fit')
        sill,range_,nugget= None,None,None
        
        if (not ('+' in model_name)) and (model_params is not None) :
            range_ = model_params['range']
            nugget =np.exp(model_params['nugget_ln'])
            sill = np.exp(model_params['sill_ln'])+nugget
            ax.vlines(range_,0,max(gamma),color="#FF1E00",linestyle='--',alpha=0.2,label=f"range")
            ax.hlines(sill,0,max(centers),color="#BD6D12",linestyle='--',alpha=0.2,label=f"sill")
        
        elif ('+' in model_name) and (model_params is not None) :
            range_ = gutils.effective_range(centers,model_fct,0.95)
            nugget= np.exp(model_params['nugget_ln'])
            sill = np.exp(model_params['sill1'])+np.exp(model_params['sill2'])+nugget
            ax.vlines(range_,0,max(gamma),color='#FF1E00',linestyle='--',alpha=0.2,label=f'effective range')
            ax.hlines(sill,0,max(centers),color='#BD6D12',linestyle='--',alpha=0.2,label=f"total sill")
        if (range_ is not None) & (sill is not None) & (nugget is not None):
            weights = None
            if counts is not None : 
                weights = gutils.get_weights_from_pair_counts(counts)
            r2 = utils.r2(gamma, model_fct(centers), weights=weights)
            textstr = f"Range: {range_:.2e} m\nSill: {sill:.2f} ‰²\nNugget: {nugget:.2f} ‰²\nR²: {r2:.2f}"
            ax.text(textbox_loc[0],textbox_loc[1], textstr,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='bottom',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8)
                    )
        if ax_ is None : plt.legend(loc=legend_loc)
    
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
    ax.set_ylabel("Semivariance [‰²]")
    
    if ax_ is None :
        plt.title(f"Empirical variogram - {time}")
        if save_name is not None:
            plt.savefig(save_name,dpi=500,bbox_inches='tight')
        return fig,ax
    return ax

def plot_elevation_map( df, col_elevation : str = 'ele', countries_borders: bool = False) -> tuple[Figure, Axes]:
    """ Plot an elevation map from given latitudes, longitudes and elevation grid.
    """
    fig, ax = plt.subplots(figsize=(10, 5),
                           subplot_kw={"projection": ccrs.PlateCarree()})

    sc = ax.scatter(df['lon'], df['lat'], c=df[col_elevation], cmap='terrain', s=2, edgecolor=None)
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

def plot_dist_to_coast_map(df,col_dist :str = 'D',countries_borders: bool = False) -> tuple[Figure, Axes]:
    """ Plot a distance to coast map from given df containing latitudes, longitudes and distance to coast grid.
    """
    fig, ax = plt.subplots(figsize=(10, 5),
                           subplot_kw={"projection": ccrs.PlateCarree()})

    sc = ax.scatter(df['lon'], df['lat'], c=df[col_dist], cmap='viridis', s=2, edgecolor=None)
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


def plot_global_map(data:pd.DataFrame,
                    title:str,
                    quantity_col:str='d18O_measurement',
                    quantity:str='d18O',
                    unit:str='‰ VPDB',
                    proj:bool=True,
                    symbol : str ='square',
                    size : int =10,
                    lon_col : str = 'longitude',
                    lat_col : str = 'latitude',
                    colorscale : str ='plasma',
                    landcolor="#fffafa",
                    oceancolor="#83d0f1")-> go.Figure:
    ''' 3D or flat earth (natural earth proj)
    If proj=True : 2D 
    else : 3D
    '''
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=data[lon_col],
        lat=data[lat_col],
        text=data[quantity_col],
        mode="markers",
        marker=dict(
            symbol=symbol,
            size=size,
            color=data[quantity_col],
            colorscale=colorscale,
            # cmin=-15.5,
            # cmax=0,
            opacity=0.7,
            # line=dict(color="white", width=1),
            colorbar=dict(
                title=f"{quantity}({unit})",
                ticks="outside",
                ticklen=6,
                thickness=15
            )
        )
    ))
    if proj :
        fig.update_layout(
            geo=dict(
                projection=dict(type="natural earth"),
                showland=True,
                landcolor=landcolor,
                showocean=True,
                oceancolor=oceancolor,
                showcountries=True,
                showcoastlines=True,
                showframe=False,
                fitbounds="locations"
            )
        )
    else :
        fig.update_layout(
            geo=dict(
                projection=dict(type="orthographic", rotation=dict(lat=12, lon=0)),
                showland=True,
                landcolor=landcolor,
                showocean=True,
                oceancolor=oceancolor,
                showcountries=True,
                showcoastlines=False,
                showframe=False
            )
        )
        # fig.write_html("../output/mean_values_d18O_interactive_map_sisal.html", include_plotlyjs="cdn")
    fig.update_layout(
        title=dict(
                text=title,
                x=0.5,
                xanchor="center",
                font=dict(size=20, family="Arial, sans-serif")
            ),
            margin=dict(r=20, l=20, t=50, b=20),
            template="plotly_white"
    )
    return fig