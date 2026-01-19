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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.axes import Axes

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
    eps = 5
    ax.set_extent([         #type:ignore
        valid[lon_col].min()-eps,
        valid[lon_col].max()+eps,
        valid[lat_col].min()-eps,
        valid[lat_col].max()+eps
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

def plot_variogram_from_bins_and_gamma(centers : np.ndarray,
                                       gamma : np.ndarray,
                                       time : str, 
                                       counts : np.ndarray | None = None, 
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
                                       ) -> tuple[Figure,Axes] | Axes | tuple:
    """ Plot an empirical variogram from given bin centers and semivariances values.
    Overlays pairs number per bin if counts is given. Does not plot bins with less than min_pairs if counts is given.
    """
    if counts is not None :
        reliable = counts >= min_pairs
        centers = centers[reliable]
        if all(~ reliable):
            print('All bins have less than {min_pairs} items, we cannot produce a reliable variogram.')
            return (None,None)
        gamma = gamma[reliable]
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

def plot_ked_platecarree_points(lon_pred, lat_pred, z_pred, ss_pred,
                                df_exp, value_col='d18Op_VSMOW',
                                title="Kriging with External Drift (PlateCarree)",
                                cmap="plasma",
                                figsize=(14,6),
                                vmin=None, vmax=None,
                                s_pred=20,  # size of predicted squares
                                s_obs=40    # size of observation points
                            ):
    """
    Plot KED kriging results in lon/lat (PlateCarree projection) using discrete scatter plots.
    Inputs :
        lon_pred, lat_pred : 1D arrays of longitudes and latitudes for predicted points
        z_pred : 1D array of predicted values
        ss_pred : 1D array of kriging variances
        df_exp : DataFrame with observed points (must have lon/lat columns)
        value_col : column name in df_exp for observed values
    """
    
    # color limits
    vmin = vmin if vmin is not None else np.nanmin(z_pred)
    vmax = vmax if vmax is not None else np.nanmax(z_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize,subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.08,
        top=0.90,
        wspace=0.12
    )
    # set extent with padding
    lon_min, lon_max = np.min(lon_pred), np.max(lon_pred)
    lat_min, lat_max = np.min(lat_pred), np.max(lat_pred)
    pad_lon = (lon_max - lon_min) * 0.05
    pad_lat = (lat_max - lat_min) * 0.05

    for ax in axes:
        ax.set_extent([lon_min - pad_lon, lon_max + pad_lon,lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
        ax.add_feature(cfeature.OCEAN, facecolor="#dff4fd", edgecolor="none")
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
        gl.right_labels = gl.top_labels = False

    sc1 = axes[0].scatter(lon_pred, lat_pred,c=z_pred,cmap=cmap,vmin=vmin, vmax=vmax,s=s_pred,marker='s', edgecolor='none',transform=ccrs.PlateCarree())
    axes[0].scatter(df_exp['lon'], df_exp['lat'],c=df_exp[value_col],cmap=cmap,vmin=vmin, vmax=vmax,s=s_obs,edgecolor='black',linewidth=0.4,transform=ccrs.PlateCarree())
    # cbar1 = plt.colorbar(sc1, ax=axes[0], orientation='vertical', shrink=0.75, pad=0.05)
    divider = make_axes_locatable(axes[0])
    cax1 = divider.append_axes("right", size="4%", pad=0.05,axes_class=Axes)
    cbar1 = fig.colorbar(sc1, cax=cax1)
    cbar1.set_label(f"{value_col} (‰ VSMOW)")
    axes[0].set_title("Kriging Prediction (discrete)")

    sc2 = axes[1].scatter(lon_pred, lat_pred,c=ss_pred,cmap='plasma',s=s_pred, marker='s',edgecolor='none',transform=ccrs.PlateCarree())
    # cbar2 = plt.colorbar(sc2, ax=axes[1], orientation='vertical', shrink=0.75, pad=0.05)
    divider = make_axes_locatable(axes[1])
    cax2 = divider.append_axes("right", size="4%", pad=0.05, axes_class=Axes)
    cbar2 = fig.colorbar(sc2, cax=cax2)
    cbar2.set_label("Kriging Variance [‰²]")
    axes[1].set_title("Kriging Variance (discrete)")

    fig.suptitle(title, fontsize=15,y=0.94)
    # plt.tight_layout()
    return fig, axes


def plot_interdistances_graph(locs_1,locs_2,title='Graph of point pairs'):
    """
    Plot the graph of pairs of points on a map, platecarree proj.    
    Inputs :

        - locs_1: array of shape (n,2) containing (longitude,latitude) coordinates of the first nodes of edges
        - locs_2: array of shape (n,2) containing (longitude,latitude) coordinates of the second nodes of edges
    """
    proj = ccrs.PlateCarree()
    
    fig, ax = plt.subplots(
        figsize=(10, 6),
        subplot_kw=dict(projection=proj)
    )
    # Map background
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)#type:ignore
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)#type:ignore
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", alpha=0.5)#type:ignore
    ax.add_feature(cfeature.OCEAN, facecolor="#dff4fd")#type:ignore

    # Plot graph edges
    for (lon1, lat1), (lon2, lat2) in zip(locs_1, locs_2):
        ax.plot(
            [lon1, lon2],
            [lat1, lat2],
            transform=ccrs.PlateCarree(),
            linewidth=0.6,
            alpha=0.4,
            color="k"
        )

    # plot nodes
    ax.scatter(
        locs_1[:, 0], locs_1[:, 1],
        s=5, color="red", transform=ccrs.PlateCarree(), zorder=3
    )
    ax.scatter(
        locs_2[:, 0], locs_2[:, 1],
        s=5, color="red", transform=ccrs.PlateCarree(), zorder=3
    )

    ax.set_title(title)

    return fig,ax
