import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
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
import matplotlib.gridspec as gridspec
from collections import OrderedDict
from matplotlib.colors import Normalize

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
    ax.set_title(f"{title} {time}")
    return fig, ax

def plot_isoscape_latlon_platecarree_df(
        df,
        time: str = '',
        title: str |None = "Values of d18O",
        countries_borders: bool = False,
        lat_col :str = 'lat',
        lon_col :str = 'lon',
        qty_col :str = 'd18Op',
        save_fp : str|None = None,
        s = 20,
        size_col : str|None = None,
        unit : str|None = '‰',
        cmap='viridis',
        cmap_sym0 = False,
        v_min : float|None = None,
        v_max : float|None = None,
        qty_label : str| None = None,
        figsize=(10,5),
        adjust_extent=True,
        landcolor = "#d8d8d8",
        oceancolor = "#dff4fd",
        fig = None,
        ax= None,
        marker='s',
    ):
    """
    Plot the given dataframe with columns: lat, lon, qty on PlateCarree projection.
    
    Parameters:
    -----------
    size_col : str|None, optional
        Column name in df containing marker sizes. If provided, marker sizes will be 
        proportional to the values in this column. If None, all markers will have size `s`.
    """
    sns.set_theme(context='talk',
                  style='ticks',
                  palette='colorblind',
                  rc={'axes.linewidth':1.2,"grid.alpha":0.3,"grid.linestyle":'--'})
    if cmap=='icefireblack':
        # colorblind 
        cmap = sns.color_palette("icefire", as_cmap=True)
    if cmap_sym0 :
        v_min = min(df[qty_col].min(),-df[qty_col].max())
        v_max = max(df[qty_col].max(),-df[qty_col].min())

    if (fig == None) or (ax == None):
        fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw={"projection": ccrs.PlateCarree()})

    if countries_borders:
        ax.add_feature(cfeature.BORDERS, linewidth=0.2)#type:ignore
    # ax.coastlines() # type:ignore
    ax.add_feature(cfeature.COASTLINE, linewidth=0.2) #type:ignore
    ax.add_feature(cfeature.LAND, facecolor=landcolor, edgecolor="none")#type:ignore
    ax.add_feature(cfeature.OCEAN, facecolor=oceancolor, edgecolor="none")#type:ignore
    ax.set_axis_off()
    # gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')#type:ignore
    # gl.right_labels = gl.top_labels = False
    # Scatter plot of irregular points
    marker_sizes = df[size_col] if size_col is not None else s
    sc = ax.scatter(
        df[lon_col],
        df[lat_col],
        c=df[qty_col],
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        s=marker_sizes,
        edgecolor=None,#"#404040",
        linewidths=0.3,
        alpha=0.99,
        vmin=v_min,
        vmax = v_max,
        marker=marker
    )

    # cbar = plt.colorbar(sc, ax=ax, label=qty_col)
    if qty_label is None :
        qty_label = qty_col
    divider = make_axes_locatable(ax)
    cax= divider.append_axes("right", size="4%", pad=0.02,axes_class=Axes)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.outline.set_visible(False) # type: ignore
    if unit is not None :   
        unit = f' [{unit}]'
    else :
        unit = ''
    cbar.set_label(qty_label+unit)

    # Add size legend to the right if size_col is specified
    if size_col is not None:
        # Create axes for the size legend to the right of the colorbar
        cbar_pos = cax.get_position()
        legend_width = cbar_pos.width * 0.22
        legend_x = cbar_pos.x0 + cbar_pos.width + 0.05
        size_legend_ax = fig.add_axes([legend_x, cbar_pos.y0, legend_width, cbar_pos.height]) #type:ignore
        
        # Get size values for reference circles (min, median, max)
        size_min = df[size_col].min()
        size_max = df[size_col].max()
        size_mid = (size_min + size_max) / 2
        
        size_values = [size_min, size_mid, size_max]
        size_labels = ['Min', 'Med', 'Max']
        
        # Plot reference circles
        for i, (size_val, label) in enumerate(zip(size_values, size_labels)):
            y_pos = 0.8 - i * 0.1
            size_legend_ax.scatter([0.26], [y_pos], s=size_val, c='gray', alpha=0.6,
                                  transform=size_legend_ax.transAxes, edgecolors='darkgray', linewidths=0.5,marker=marker)
            size_legend_ax.text(0.33, y_pos, f'{label} ({size_val/100:.2f})', ha='left', va='center',
                               transform=size_legend_ax.transAxes, fontsize=16)
        
        # Add title for the size legend
        size_legend_ax.text(0.26, 0.89, r'Size : $\sigma(\text{PES})$', ha='left', va='top',
                           transform=size_legend_ax.transAxes, fontsize=16, weight='bold')
        
        # Clean up the axes
        size_legend_ax.set_xlim(0, 1)
        size_legend_ax.set_ylim(0, 1)
        size_legend_ax.axis('off')

    valid = df.dropna(subset=[lat_col, lon_col, qty_col])
    eps = 3
    if adjust_extent :
        ax.set_extent([    #type:ignore
            valid[lon_col].min()-eps,
            valid[lon_col].max()+eps,
            valid[lat_col].min()-eps,
            valid[lat_col].max()+eps
        ], crs=ccrs.PlateCarree())
    else :
        ax.set_extent([-180,180,-90,90]) #type:ignore
    if title is not None :
        ax.set_title(f"{title}{time}")
    plt.tight_layout()
    if save_fp is not None :
        plt.savefig(save_fp,dpi=400)
    else : 
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

def plot_variogram_from_bins_and_gamma(bin_edges : np.ndarray,
                                       gamma : np.ndarray,
                                       title : str, 
                                       counts : np.ndarray | None = None, 
                                       std_counts=None,
                                       plot_model : bool = True, 
                                       model_name : str|None = None, 
                                       model_fct = None,
                                       model_params = None,
                                       figsize: tuple[int,int]=(15,5),
                                       textbox_loc = (0.05,0.4),
                                       ax_ = None,
                                       save_name : str | None = None,                                      
                                       ) -> tuple[Figure,Axes] | Axes | tuple:
    """ Plot an empirical variogram from given bin centers and semivariances values.
    Overlays pairs number per bin if counts is given. Does not plot bins with less than min_pairs if counts is given.
    """
    # Seaborn theme
    sns.set_theme(context='talk',
                  style='ticks',
                  palette='colorblind',
                  rc={'axes.linewidth':1.2})
    
    # Define bin centers :
    bin_centers = bin_edges - np.r_[bin_edges[0],np.diff(bin_edges)]/2

    # Create the fig/ax on which to plot
    if ax_ is None:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            nrows=1,
            ncols=2,
            width_ratios=[3.2, 1.2],
            wspace=0.3
        )
        ax = fig.add_subplot(gs[0, 0])
        ax_info = fig.add_subplot(gs[0, 1])
    else:
        ax = ax_
        ax_info = None
    colors=sns.color_palette('colorblind')
    
    # Empirical vario 
    ax.scatter(bin_centers,
            gamma,
            marker='+',
            # linewidth=1.5,
            # linestyle='--',
            s=50,
            c='black',#colors[0],
            label='Empirical variogram',
            zorder=3)
    
    # Overlay the vario model is given
    textstr = None
    if (plot_model) and (model_name is not None) and (model_fct is not None):
        h = np.linspace(0,bin_edges.max(),300)
        ax.plot(h,
                model_fct(h),
                linestyle='-',
                color=colors[1],
                linewidth=2,
                label=f'{model_name} fit',
                zorder=4)
        # retrieve the fit parameters for drawing associated lines and legend
        sill,range_,nugget= None,None,None
        if model_params is not None :
            n_models = len(model_name.split('+'))
            if n_models ==1 :
                if model_name != 'nugget' :
                    range_ = model_params[f'range_{model_name}']
                    sill = model_params[f'sill_{model_name}']
                    ax.axvline(range_,color=colors[2],linestyle='--',alpha=0.3,label="Range")
                    ax.axhline(sill,color=colors[3],linestyle='--',alpha=0.3,label="Sill")
                if model_name == 'nugget' :
                    nugget = model_params['nugget']
                    ax.axhline(nugget,color=colors[4],linestyle='--',alpha=0.3,label="Nugget")
            elif n_models ==2 and model_name.count('nugget')==1:
                # Get the name of the non-nugget model
                non_nugget_model = [m for m in model_name.split('+') if m != 'nugget'][0]
                range_ = model_params[f'range_{non_nugget_model}']
                sill = model_params[f'sill_{non_nugget_model}']+model_params['nugget']
                ax.axvline(range_,color=colors[2],linestyle='--',alpha=0.3,label="Range")
                ax.axhline(sill,color=colors[3],linestyle='--',alpha=0.3,label="Sill")
                nugget = model_params['nugget']
            else:
                range_ = gutils.effective_range(bin_edges,model_fct,0.95)
                nugget=0.0
                sill=0.0
                if 'nugget' in model_name:
                    nugget = model_params['nugget']
                prev_models=[] # handle the case of nested models of the same type like spherical+spherical
                for model in model_name.split('+'):
                    if model!='nugget':
                        if model in prev_models:
                            k = prev_models.count(model)+1 # because i put a number in the params if there are multiple models
                            sill += model_params[f'sill_{model}_{k}']
                        else :
                            sill += model_params[f'sill_{model}']
                            prev_models.append(model)
                    else : 
                        sill += nugget
                ax.axvline(range_,color=colors[2],linestyle='--',alpha=0.3,label='Effective range')
                ax.axhline(sill,color=colors[3],linestyle='--',alpha=0.3,label="Total sill (+nugget)")
        if (ax_info is not None):
            weights = None
            if counts is not None : 
                weights = gutils.get_weights_from_pair_counts(counts)
            r2 = utils.r2(gamma, model_fct(bin_edges), weights=weights)
            textstr=''
            if range_ is not None :
                textstr += rf"$\bf{{Range:}}$ {range_:.2e}m""\n"
            if sill is not None :
                textstr += rf"$\bf{{Total~sill:}}$ {sill:.2f}‰$^2$""\n"
            if nugget is not None :
                textstr += rf"$\bf{{Nugget:}}$ {nugget:.2f}‰$^2$""\n"
            if r2 is not None :
                textstr += rf"$\bf{{R^2:}}$ {r2:.2f}"
    # Compute widths :
    widths = np.diff(bin_edges)
    widths = np.r_[bin_edges[0],widths]

    # Overlay bins counts if given
    if counts is not None:
        ax2 = ax.twinx()
        ax2.bar(bin_centers,
                counts,
                width = widths,
                color=colors[0],
                alpha=0.15,
                align='center',
                label='Pair counts',
                zorder=1)
        ax2.set_ylabel('Number of pairs',color=colors[0])
        ax2.tick_params(axis='y', colors=colors[0])   
        ax2.spines['right'].set_color(colors[0]) 
        ax2.grid(False)
        if std_counts is not None :
            ax2.bar(bin_centers,
                    std_counts,
                    width = widths,
                    color=colors[1],
                    alpha=0.15,
                    label='Pair counts std',
                    zorder=1)
    # axes format
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    # ax.grid(True, alpha=0.4)
    ax.set_xlabel("Interdistance [m]")
    ax.set_ylabel("Semivariance [‰²]")
    sns.despine(ax=ax, top=True, right=True)
    if counts is not None :
        sns.despine(ax=ax2, top=True, right=False)

    if ax_info is not None:
        if counts is not None:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            handles = h1 + h2
            labels = l1 + l2
        else:
            handles, labels = ax.get_legend_handles_labels()
        # Remove duplicates 
        by_label = OrderedDict(zip(labels, handles))
        ax_info.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper left",
            frameon=False,
            fontsize=12,
        )
        if textstr is not None :
            ax_info.text(textbox_loc[0],textbox_loc[1], textstr, # type:ignore
                        transform=ax_info.transAxes,
                        fontsize=12,
                        va='bottom',
                        ha='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
                        )
        ax_info.axis('off')
        
    if ax_ is None :
        plt.title(title)
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
                    unit:str='‰',
                    proj:bool=True,
                    symbol : str ='square',
                    size : int =10,
                    lon_col : str = 'longitude',
                    lat_col : str = 'latitude',
                    colorscale : str ='plasma',
                    landcolor="#fffafa",
                    oceancolor="#83d0f1",
                    save_fig : str|None =None)-> go.Figure:
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
    if save_fig is not None:
        fig.write_html(save_fig, include_plotlyjs="cdn")
    return fig

def plot_platecarree_field_and_scatter_on_top(field_df,
                                              points_df,
                                              lat='lat',
                                              lon='lon',
                                              qty_field='d18Op',
                                              qty_points='d18Op',
                                              colorbarlabel=r'$\delta^{18}\text{O}_p$',
                                              title=r"External drift $\delta^{18}\text{O}_p$ (square markers) and its values at obs points (circles)",
                                              vmin = -30,
                                              vmax = 0,
                                              figsize=(10,6),
                                              markersize=30,
                                              ):
    """ This function plots a field overlays a scatter plot on top of it. Designed for checking the values of a qty (qty_points) at some locs of the field.
    """
    # Normalization range
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})

    # Plot the field values as colored squares
    scatter_field = ax.scatter(
        field_df[lon],
        field_df[lat],
        c=field_df[qty_field],
        cmap='plasma',
        norm=norm,
        s=markersize,
        marker='s',         # Square markers
        edgecolor='none',
        transform=ccrs.PlateCarree()
    )

    # Add colorbar
    plt.colorbar(scatter_field, ax=ax, label=colorbarlabel)

    # Overlay the observation points
    ax.scatter(
        points_df[lon],
        points_df[lat],
        c=points_df[qty_points],
        cmap='plasma',
        norm=norm,
        s=markersize+20,
        edgecolor='k',
        transform=ccrs.PlateCarree()
    )

    ax.coastlines() # type:ignore
    ax.set_title(title)
    return fig,ax

    
def plot_ked_platecarree_points(df_pred,
                                df_obs, value_col='d18Op_VSMOW',
                                title=None,
                                cmap="plasma",
                                figsize=(20,6),
                                vmin=None, vmax=None,
                                lon='lon',
                                lat='lat',
                                s_pred=20,  # size of pred squares
                                s_obs=40 ,   # size of obs points,
                                adjust_extent:bool=True
                            ):
    """
    Plot KED kriging results in lon/lat (PlateCarree projection) using discrete scatter plots.
    Inputs :
        df_pred : DataFrame with predicted points (must have lon/lat columns)
        df_obs : DataFrame with observed points (must have lon/lat columns)
        z_pred : 1D array of predicted values
        ss_pred : 1D array of kriging variances
        df_exp : DataFrame with observed points (must have lon/lat columns)
        value_col : column name in df_exp for observed values
    """
    # color limits
    vmin = vmin if vmin is not None else np.nanmin(df_pred['z_pred'])
    vmax = vmax if vmax is not None else np.nanmax(df_pred['z_pred'])

    fig, axes = plt.subplots(1, 2, figsize=figsize,subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.08,
        top=0.99,
        wspace=0.2
    )

    for ax in axes:
        if adjust_extent :
            ax.set_extent([df_pred[lon].min(), df_pred[lon].max(), df_pred[lat].min(), df_pred[lat].max()], crs=ccrs.PlateCarree())#[lon_min - pad_lon, lon_max + pad_lon,lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.2)
        ax.add_feature(cfeature.LAND, facecolor="#d8d8d8", edgecolor="none")
        ax.add_feature(cfeature.OCEAN, facecolor="#dff4fd", edgecolor="none")
        # gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
        # gl.right_labels = gl.top_labels = False

    sc1 = axes[0].scatter(df_pred[lon],
                          df_pred[lat],
                          c=df_pred['z_pred'],
                          cmap=cmap,vmin=vmin, vmax=vmax,s=s_pred,marker='s', edgecolor='none',transform=ccrs.PlateCarree())
    axes[0].scatter(df_obs[lon],
                    df_obs[lat],
                    c=df_obs[value_col],
                    cmap=cmap,vmin=vmin, vmax=vmax,s=s_obs,edgecolor='black',linewidth=0.4,transform=ccrs.PlateCarree())
    # cbar1 = plt.colorbar(sc1, ax=axes[0], orientation='vertical', shrink=0.75, pad=0.05)
    divider = make_axes_locatable(axes[0])
    cax1 = divider.append_axes("right", size="4%", pad=0.05,axes_class=Axes)
    cbar1 = fig.colorbar(sc1, cax=cax1)
    cbar1.set_label(r"$\delta^{18}\text{O}_p$ [‰]", fontsize=20)
    # axes[0].set_title("Kriging Prediction", fontsize=16)

    sc2 = axes[1].scatter(df_pred[lon],
                          df_pred[lat],
                          c=df_pred['ss_pred'],
                          cmap='plasma',s=s_pred, marker='s',edgecolor='none',transform=ccrs.PlateCarree())
    # cbar2 = plt.colorbar(sc2, ax=axes[1], orientation='vertical', shrink=0.75, pad=0.05)
    divider = make_axes_locatable(axes[1])
    cax2 = divider.append_axes("right", size="4%", pad=0.05, axes_class=Axes)
    cbar2 = fig.colorbar(sc2, cax=cax2)
    cbar2.set_label(r"$\sigma_\text{KED}$ [‰$^2$]", fontsize=20)
    # axes[1].set_title("Kriging Variance", fontsize=16)

    if title is not None: 
        fig.suptitle(title, fontsize=18,y=0.94)
    return fig,ax


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
    ax.scatter(locs_1[:, 0], locs_1[:, 1],s=5, color="red", transform=ccrs.PlateCarree(), zorder=3)
    ax.scatter(locs_2[:, 0], locs_2[:, 1],s=5, color="red", transform=ccrs.PlateCarree(), zorder=3)
    ax.set_title(title, fontdict={'fontsize':12})

    return fig,ax


def plot_scatter_with_ci(df,alpha,fp,figsize=(10,10)):
    """ Plots a scatter plot of prediction points with a simulatenous confidence interval of (semi)length ci, and also the truth value.
    df must contain columns z_pred,z_obs , ci. Argument alpha is the maximum acceptable type 1 error rate used in the statistical test. 
    Saves the figure at path fp
    """
    sns.set_theme(context='talk',
                  style='ticks',
                  palette='colorblind',
                  rc={'axes.linewidth':1.2,"grid.alpha":0.3,"grid.linestyle":'--'})
    
    fig,ax = plt.subplots(figsize=figsize)

    ax.scatter(df.z_obs,df.z_pred,zorder=3,label='LOO predictions')
    ax.errorbar(df.z_obs,df.z_pred,yerr=df.ci,fmt='none',capsize=3,elinewidth=0.5,zorder=2,label=f"{int((1-alpha)*100)}% simultaneous  CI.")
    lims = [min(df.z_obs.min(),df.z_pred.min()),max(df.z_obs.max(),df.z_pred.max())]
    ax.plot(lims,lims,linestyle='--',linewidth=1, label="x=y")
    ax.set_xlabel('Observed values')
    ax.set_ylabel('LOO kriging prediction')
    ax.legend()
    # ax.set_aspect('equal',adjustable='box')
    ax.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(fp,dpi=400)