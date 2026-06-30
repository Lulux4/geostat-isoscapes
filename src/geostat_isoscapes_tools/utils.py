import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np 
import shapely 
from pyproj import CRS
from scipy.spatial.distance import cdist
import subprocess
import os
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from scipy.linalg import lstsq
from sklearn.neighbors import BallTree

# ==========================================
# ROOT DIR RETRIEVAL
# ==========================================
def get_project_root():
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    while True:
        if os.path.exists(os.path.join(current_dir, "pyproject.toml")):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError("Project root (with pyproject.toml) not found.")
        current_dir = parent_dir

# =========================================================================================
# MISC.TOOLS FOR DATA LOADING, BINNING
# =========================================================================================
def load_xarray_datarray(fn: str) -> xr.Dataset:
    """ Just a function to open a netcdf dataset """
    with xr.open_dataset(filename_or_obj=fn, engine='netcdf4',decode_times=False) as file:
        return file

def prepare_ds_of_ked_isoscsape(ked_df : pd.DataFrame)->xr.Dataset :
    ds = ked_df.set_index(['time', 'lat', 'lon']).to_xarray()
    ds['d18Op'].attrs.update({
        'units': 'per mil',
        'description': 'd18Op VSMOW obtained by KED interpolation from SISAL speleothem dataset and iTrace simulations (assumed as external drift).',
    })
    ds['ss_pred'].attrs.update({
        'units': 'per mil squared',
        'description': 'Kriging variance of the d18Op VSMOW after KED interpolation.',
    })
    ds.coords['lon'].attrs.update({'units': 'degrees', 'description': 'Longitude, from -180 degrees (W180) to +180 degrees (E180).'})
    ds.coords['lat'].attrs.update({'units': 'degrees', 'description': 'Latitude, from -90 degrees (S90) to +90 degrees (N90).'})
    ds.coords['time'].attrs.update({'units': 'years', 'description': 'Years before Present (1950). Positive integers.'})
    
    # add the information on the within cell altitude variance :
    std_elev_da = xr.open_dataarray(f"{get_project_root()}/output/std_elev.nc")
    std_elev_da, ds = xr.align(
        std_elev_da,
        ds,
        join="inner",   # keep only the overlapping coordinates
        copy=False,
    )
    ds = ds.assign(std_elev=std_elev_da)
    ds["std_elev"].attrs["units"] = "m"
    ds["std_elev"].attrs["description"] = ("standard deviation of the within-cell elevation computed from a digital elevation model at 1 arc-minute resolution")
    
    return ds 


def slice_in_equal_bins(series : pd.Series, bin_width: int) -> pd.Series :
    """ This function bins a pandas Series with integer bin labels according to the given bin width.
    
    This function outputs the series of bins labels, which are defined as the upper bound of each bin range. 
    E.g : bin "width" spans [0,0+width[.

    If the series has positive and negative values, the binning forces the creation of bins from 0 to the max, and from 0 to the min in the reverse direction.
    Ex : [-19,-16,-4,-1,2,6,8,18] with a bin width of 5 is binned as [-20,-20,-5,0,5,10,10,20].
    
    Inputs :
        - series : pd.Series containing float values to bin
        - bin_width : integer of the desired bin width 
    Outputs : 
         - binned_series : pd.Series containing the bin label associated of each item. 
    """
    bins=None
    if series.max() > 0 :
        start = max(0,series.min())
        bins_pos = np.array(range(int(start-bin_width), int(series.max())+ bin_width + 1, bin_width))
        bins = list(bins_pos)
    if series.min()<0 :
        start = min(0,(series.max()//3+1) *3)
        stop = min(series.min()-bin_width,start-bin_width)
        bins_neg = np.sort(- np.array(range( - int(start), -int(stop), bin_width)))
        if bins is not None : 
            bins = list(np.sort(list(set(list(np.concat([bins_neg,bins_pos]))))))
        else :
            bins = list(bins_neg)
    
    if bins is None : # sanity check
        raise ValueError('this series does not contain numbers or is empty.')

    binned_series = pd.cut(series, bins, labels = bins[1:],right=True)
    return binned_series

def bin_xrDataArray_time(da : xr.DataArray, res : int, method : str = 'median', weights : xr.DataArray | None = None) -> xr.DataArray : 
    """ This function bins a xs dataarray time at the specified resolution and applies a median on the dataarray attributes values."""
    tmin, tmax = float(da.time.min()), float(da.time.max())
    bins = np.arange(tmin, tmax + int(res), int(res)) #type:ignore

    if method == 'median':
        da_binned = da.groupby_bins('time', bins=bins).median()
    elif method == 'sum':
        da_binned = da.groupby_bins('time', bins=bins).sum()
    elif method == 'weighted_mean' :
        if weights is None :
            raise ValueError("Weights must be provided for weighted mean method.")
        # Compute weighted mean using xarray operations: sum(da * weights) / sum(weights)
        da_weighted = (da * weights).groupby_bins('time', bins=bins).sum()
        sum_weights = weights.groupby_bins('time', bins=bins).sum()
        da_binned = da_weighted / sum_weights
    else:
        raise ValueError("Invalid method. Choose 'median', 'sum', or 'weighted_mean'.")

    da_binned = da_binned.rename({'time_bins': 'time'})
    return da_binned.assign_coords(time=bins[:-1])

def get_yrBP_from_itrace_time(months_after_start : int | pd.Series | np.ndarray, start_year : float = 12001)-> float| np.ndarray| pd.Series:
    """ Retrieve the year (+decimals) depending on a start year in yr BP (positive number) and the number of months spent since this start year.
    """
    return -( -start_year + months_after_start/12)

def set_itrace_xarray_time_to_yrBP(da : xr.DataArray | xr.Dataset, start_year : float = 12001) -> xr.DataArray | xr.Dataset:
    """ This function sets the time coordinate of a xarray dataset or dataarray to years before present (1950), depending on a start year in yr BP and the number of months spent since this start year. 
    The time coordinate is expected to be in months after the start year, as in iTrace simulations.
    """
    timearray = get_yrBP_from_itrace_time(np.array(range(0,len(da.time),1)), start_year)
    da = da.assign_coords(time=('time',-timearray)) # type:ignore # we keep the minus sign to conserve the order 20 ka BP before 19 ka BP (for the binning)
    da = da.assign_coords(time = da.time.assign_attrs(units=f"years before present (1950)"))
    return da

def define_d18O_itrace(rain_h2otr_fp,rain_h218o_fp,snow_h2otr_fp,snow_h218o_fp,kyr):
    """ This function defines the d18O of precipitation in iTrace simulations, using the RAIN_H2OTR, RAIN_H218O, SNOW_H2OTR and SNOW_H218O variables. 
    The d18O is defined as : d18O = ((RAIN_H218O + SNOW_H218O) / (RAIN_H2OTR + SNOW_H2OTR) -1)*1000
    """
    file_rH218O = load_xarray_datarray(rain_h218o_fp)
    file_rH2OTR = load_xarray_datarray(rain_h2otr_fp)
    file_sH218O = load_xarray_datarray(snow_h218o_fp)
    file_sH2OTR = load_xarray_datarray(snow_h2otr_fp)

    # compute the delta18Op at each time step and each point
    h218o = file_rH218O.RAIN_H218O
    h2o = file_rH2OTR.RAIN_H2OTR
    h218o += file_sH218O.SNOW_H218O
    h2o += file_sH2OTR.SNOW_H2OTR
    delta18 = ( h218o/h2o - 1.0) * 1000.0
    delta18 = delta18.where( (h2o> 1e-12) & (delta18 < 1e2) )  # avoid div by near-0 precip values and large positive outliers (delta18 should be mostly negative and small -20/+20)

    # The itrace doc specifies explicetely that the temporal resolution is in months, so we can define the time dimension in terms of months after a start year.
    # timearray = range(0,len(delta18.time),1)
    # delta18 = delta18.assign_coords(time=('time',timearray))
    # delta18 = delta18.assign_coords(time = delta18.time.assign_attrs(units=f"months since start year ({kyr} ka)"))
    delta18 = set_itrace_xarray_time_to_yrBP(delta18, start_year=kyr*1000)
    delta18 = delta18.astype('float64').assign_coords(lat=delta18.lat.astype('float64').round(3))

    return delta18

def define_precipitation_itrace(rain_fp,snow_fp,kyr):
    """ This function defines the precipitation amount in iTrace simulations, using the RAIN and SNOW variables.
    """
    file_rain = load_xarray_datarray(rain_fp)
    file_snow = load_xarray_datarray(snow_fp)
    precip_da = file_rain.RAIN
    precip_da += file_snow.SNOW

    # TODO : i need to adapt the entire pipeline to this new logic with the time unit : i shoudl remove the commented lines when i am sure it works :)
    # The itrace doc specifies explicetely that the temporal resolution is in months, so we can define the time dimension in terms of months after a start year.
    # timearray = range(0,len(precip_da.time),1)
    # precip_da = precip_da.assign_coords(time =('time',timearray))
    # precip_da = precip_da.assign_coords(time = precip_da.time.assign_attrs(units=f"months since start year ({kyr} ka)"))
    precip_da = precip_da * 31 * 24 * 3600 # integrate over bin width (natural binwidth is 31 days)
    # Switch the time dimension to yrBP
    precip_da = set_itrace_xarray_time_to_yrBP(precip_da, start_year=kyr*1000)
    precip_da = precip_da.astype('float64').assign_coords(lat=precip_da.lat.astype('float64').round(3))

    return precip_da

def define_temperature_itrace(trefht_fp,kyr):
    """ This function defines the temperature at the surface in iTrace simulations, using the TREFHT variable.
    """
    trefht_da = load_xarray_datarray(trefht_fp).TREFHT
    # Switch the time dimension to yrBP
    trefht_da = set_itrace_xarray_time_to_yrBP(trefht_da, start_year=kyr*1000)  
    trefht_da = trefht_da.astype('float64').assign_coords(lat=trefht_da.lat.astype('float64').round(3))

    return trefht_da

def define_ele_itrace(phis_fp,kyr):
    """ This function define the elevation at the surface in iTraCE simulations using the PHIS variable (geopotential)"""
    phis_da = load_xarray_datarray(phis_fp)
    ele_da = phis_da.PHIS / 9.81
    ele_da = set_itrace_xarray_time_to_yrBP(ele_da, start_year=kyr*1000)
    ele_da = ele_da.astype('float64').assign_coords(lat=ele_da.lat.astype('float64').round(3))
    return ele_da

# =============================================================================================
# "Spatial" computations
# =============================================================================================
# def split_antimeridian(geom):
#     antimeridian = shapely.LineString([(180, -90), (180, 90)])
#     try:
#         return shapely.ops.split(geom, antimeridian)
#     except Exception:
#         return geom
    
def mask_regions_shape(da : xr.DataArray | xr.Dataset | pd.DataFrame, 
                       regions: tuple[str,list[str]]= ('continents',["Asia"]), 
                       all_touched : bool = True, 
                       shapefile : str = f"{get_project_root()}/data/shapefiles/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp",
                       buffer_km : float = 0, 
                       ) -> xr.DataArray | xr.Dataset | pd.DataFrame :
    """ This functions defines a mask for the given regions and filters the points of the provided data array
    (which should have coords lon and lat) according to this mask. If all_touched is set to True (default), the mask 
    includes pixels that have at least one corner inside the region, otherwise it includes only pixels with their center
    inside the region.
    """
    # read country borders from shapefile
    world = gpd.read_file(shapefile)
    if regions[0]=='countries':
        countries = regions[1]
    elif regions[0]=='continents':
        countries = world.loc[world['CONTINENT'].isin(regions[1]),'NAME'].to_list()
    elif regions[0]=='subregions':
        countries = world.loc[world['SUBREGION'].isin(regions[1]),'NAME'].to_list()
    else :
        raise ValueError("The first element of tuple 'regions' must be 'countries', 'subregions', or 'continents'. \n It specifies what is contained in the second element : the list of geographical names.")
    shape = world[world["NAME"].isin(countries)].to_crs("EPSG:4326")
    shape["geometry"] = shape.geometry.make_valid()
    geom = shape.union_all()
    
    # Apply buffer if requested
    if buffer_km > 0:
        # Project to equal-area or metric CRS (Asia-centered)
        shape_m = shape.to_crs(CRS.from_epsg(3857))  # Web Mercator, meters
        geom_m = shape_m.union_all()
        geom_m_buffered = geom_m.buffer(buffer_km * 1000)  # buffer in meters
        # Reproject back to lon/lat
        geom = gpd.GeoSeries([geom_m_buffered], crs=3857).to_crs(4326).iloc[0]
        geom = shapely.make_valid(geom) # type:ignore
        # geom = split_antimeridian(geom) # this does not work : todo : find how to deal with shapes that cross the antimeridian line

    if isinstance(da, xr.DataArray) or isinstance(da, xr.Dataset):
        # grid
        lat, lon = da.lat.values, da.lon.values
        
        # if longitudes are defined on the interval 0° to 360°, convert it to -180° to 180°
        if any(lon>180):
            lon = convert_lon_0_360_to_neg180_180(lon)
        if any(lat>90):
            lat = convert_lat_0_180_to_neg90_90(lat)

        lon2d, lat2d = np.meshgrid(lon,lat) #type:ignore
        # def mask
        if all_touched :
            dx = abs(lon[1] - lon[0])/2
            dy = abs(lat[1] - lat[0])/2
            mask = np.zeros_like(lon2d, dtype=bool)
            # Check if any of the 4 corners of each pixel is inside the desired polygone
            for dlon in (-dx, dx):
                for dlat in (-dy, dy):
                    corners = shapely.points(np.column_stack((lon2d.ravel() + dlon, lat2d.ravel() + dlat)))
                    mask = mask | shapely.contains(geom, corners).reshape(lon2d.shape)
        else :
            mask = shapely.contains_xy(geom, lon2d, lat2d)
        
        masked_da = da.where(mask)
        return masked_da
    
    elif isinstance(da, pd.DataFrame):
        if not {'lon', 'lat'}.issubset(da.columns):
            raise ValueError("DataFrame must contain lon/lat columns")
        points = shapely.points(da[['lon', 'lat']].to_numpy())
        mask = shapely.contains(geom, points)
        return da.loc[mask].reset_index(drop=True) #type:ignore
    
    else:
        raise TypeError("Input must be an xarray.DataArray, xr.Dataset or pandas.DataFrame")

def convert_lon_0_360_to_neg180_180(lon:np.ndarray) ->np.ndarray:
    """ Converts longitudes in the interval O to 360 degreees to -180 to 180 degrees"""
    return (lon + 180) % 360 -180

def convert_lon_neg180_180_to_0_360(lon:np.ndarray) ->np.ndarray:
    """ Converts longitudes in the interval -180 to 180 degreees to 0 to 360 degrees"""
    return lon % 360

def convert_lat_0_180_to_neg90_90(lat:np.ndarray) ->np.ndarray:
    """ Converts latitudes in the interval 0 to 180 degreees to -90 to 90 degrees"""
    return (lat + 90) % 180 -90

def haversine(u, v):
    """Haversine distance (km) between two (lat,lon) points."""
    lat1, lon1 = np.radians(u)
    lat2, lon2 = np.radians(v)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(a)) # since earth radius = approx 6371 km

def dist_to_nn_on_sphere(points):
    ''' This function computes the distance of each location in the array _points_ to its nearest neighbourg, using coordinates on a sphere'''
    coords_radians = np.deg2rad(points[["lat","lon"]].values)
    tree = BallTree(coords_radians,metric='haversine')
    dists,inds = tree.query(coords_radians,k=2)
    return dists[:,1] * 6371.0 #km

def canonical_lines(lines):
    """ Lines is an array of shape N*2*2 (N lines consituted by a start point and an end point, each with 2 coordinates).
    This function makes sure to order the end point and start points so that we can actually compare two sets of can
    """
    # lines: (N, 2, 2)
    out = lines.copy()
    swap = (
        (lines[:, 0, 0] > lines[:, 1, 0]) |
        ((lines[:, 0, 0] == lines[:, 1, 0]) &
         (lines[:, 0, 1] > lines[:, 1, 1]))
    )
    out[swap, 0], out[swap, 1] = out[swap, 1], out[swap, 0]
    return out

def mask_union_of_circles_around_pts(data_to_mask : pd.DataFrame | xr.Dataset | xr.DataArray,
                                     df_ref_pts : pd.DataFrame, 
                                     radius_km : int,
                                     lat_name="lat", 
                                     lon_name="lon",
                                     verbose:bool = False):
    """ Masks points in dataframe or xarray dataset, keeping only the union of circles centered on df_ref_pts points, with a radius radius_km. 
    """

    # get the list of (unique) ref points locations 
    unique_locs = df_ref_pts[[lat_name,lon_name]].drop_duplicates()
    ref_locs = unique_locs.values
    # ===============
    # pandas version
    # ===============
    if isinstance(data_to_mask,pd.DataFrame):
        if any(data_to_mask[lat_name]>90):
            data_to_mask[lat_name] = convert_lat_0_180_to_neg90_90(np.array(data_to_mask[lat_name]))
        if any(data_to_mask[lon_name]>180):
            data_to_mask[lon_name] = convert_lon_0_360_to_neg180_180(np.array(data_to_mask[lon_name]))

        # data points of the df to mask
        locs_to_mask = data_to_mask[[lat_name,lon_name]].drop_duplicates().reset_index(drop=True)
        # Compute min distance from each point to nearest ref point
        if verbose : print('> computing distances between data points and anchor points')
        distances = cdist(locs_to_mask[[lat_name,lon_name]].values, ref_locs, metric=haversine)  # shape (n_unique_to_mask, n_ref_pts)
        # Mask df
        locs_to_mask['mask'] = distances.min(axis=1)  <= radius_km
        if verbose : print('> done')
        return apply_spatial_mask(df=data_to_mask,spatial_mask=locs_to_mask,lat=lat_name,lon=lon_name,mask_col='mask'), locs_to_mask
    # ===============
    # xr version 
    # ===============
    elif isinstance(data_to_mask,xr.Dataset):
        # points to mask (extraction)
        lats = data_to_mask[lat_name].values
        lons = data_to_mask[lon_name].values
        if max(lons)>180:
            ref_locs[:,1]=convert_lon_neg180_180_to_0_360(ref_locs[:,1])
        lon2d, lat2d = np.meshgrid(lons,lats)
        locs = np.column_stack([lat2d.ravel(), lon2d.ravel()])  # (nlat*nlon, 2)
        # compute dist
        if verbose : print('> computing distances between data points and anchor points')
        distances = cdist(locs, ref_locs, metric=haversine)  # (npts, Nref)
        min_dist = distances.min(axis=1).reshape(lat2d.shape)  # reshape to (lat,lon)
        # mask
        mask2d = min_dist <= radius_km    # (lat, lon)
        mask = xr.DataArray(mask2d,
                            dims=(lat_name, lon_name),
                            coords={lat_name: data_to_mask[lat_name], lon_name: data_to_mask[lon_name]},
                        )#.expand_dims(time=data_to_mask.time)
        if verbose : print('> done')
        return data_to_mask.where(mask.expand_dims(time=data_to_mask.time)), mask


def apply_spatial_mask(df, spatial_mask, lat='lat', lon='lon',mask_col='mask'):
    """ Applies a mask defined on specific lat lon coordinates to a dataframe """
    merged = df.merge(spatial_mask, on=[lat, lon], how="left")    
    return merged[merged[mask_col]]
# =========================================================================================
# STATISTICAL METRICS
# =========================================================================================
def gvif(X_block : np.ndarray,X_rest : np.ndarray) :
    """ Generalized Variance Inflation Factor (Fox & Monette 1992)
    Scaled by 1/(2p) where p is the number of columns in the block of interest, in order
    to compare GVIF value to other VIF values (d=1 GVIF=VIF)
    """
    # get number of predictors on the block of interest
    p = X_block.shape[1]
    # standardize columns 
    X1 = standardize_matrix(X_block,cols=True)
    X2 = standardize_matrix(X_rest,cols=True)

    # compute (partial) correlation matrices
    R11 = np.array(np.corrcoef(X1, rowvar=False), ndmin=2) # if X_block is 1D, R11 = 1.0 --> R11 = [[1.0]]
    R22 = np.array(np.corrcoef(X2, rowvar=False), ndmin=2) # if X_rest is 1D, R22 = 1.0 --> R22 = [[1.0]]
    R = np.corrcoef(np.append(X1, X2, axis=1), rowvar=False)
    # compute gvif
    gvif = np.linalg.det(R11) * np.linalg.det(R22) / np.linalg.det(R)
    gvif = gvif ** (1/(2*p))
    return gvif

def standardize_matrix(X,cols=True):
    """ Simple function to standardize matrix columns """
    if cols :
        ax = 0
    else :
        ax = 1
    return (X - X.mean(axis=ax)) / np.sum(X,axis = ax)

def partial_r2_block(y,X_full,X_rest):
    """ Computes partial r2 by block of predictors
    X_rest = X_full except the block 
    y = response variable
    """
    # standardize columns 
    y = standardize_matrix(y,cols=True)
    X_full = standardize_matrix(X_full,cols=True)
    X_rest = standardize_matrix(X_rest, cols=True)
    
    # fit models 
    model_full = sm.OLS(y,X_full).fit()
    model_reduced=sm.OLS(y,X_rest).fit()
    r2_full = model_full.rsquared
    r2_reduced = model_reduced.rsquared
    return (r2_full-r2_reduced)/(1.-r2_reduced)

def r2(y_true : np.ndarray, y_pred : np.ndarray, weights = None) -> float:
    """ Computes the R^2 between true and predicted values 
    """
    if weights is None : 
        weights = np.ones_like(y_true)
    weighted_mean = np.nansum(weights * y_true) / np.nansum(weights)
    sst = np.nansum(weights * (y_true - weighted_mean)**2)
    ssr = np.nansum(weights * (y_true - y_pred)**2)
    r2 = 1 - (ssr / sst)
    return r2

def r2_adj(y_true : np.ndarray, y_pred : np.ndarray, p : int, weights = None):
    """ Compute the r2 adjusted for the number of parameters in the prediction model """
    if weights is None : 
        weights = np.ones_like(y_true)
    n = len(y_true)
    weighted_mean = np.nansum(weights * y_true) / np.nansum(weights)
    sst = np.nansum(weights * (y_true - weighted_mean)**2)
    ssr = np.nansum(weights * (y_true - y_pred)**2)
    adj_r2 = 1 - (ssr/(n-p-1)) / (sst/(n-1))
    return adj_r2

def RSE(y_true,y_pred,n_params):
    """ compute residual standard error of predicted values, given the number of fit parameters n_params"""
    rss = np.nansum((y_true-y_pred)**2)
    df = len(y_true)-n_params 
    return np.sqrt(rss/df)

def RMSE(y_true,y_pred):
    """ Root mean squared error of predicted values y_pred """
    return np.sqrt(np.nanmean((y_true-y_pred)**2))

def MAE(y_true,y_pred):
    """ compute mean absolute error of the predicted values """
    return np.nanmean(np.abs(y_true-y_pred))

def PES(z_obs,z_pred,ss_pred):
    """ Studentized prediction error (PES) of kriging predictions.
    - z_obs is the measured value
    - z_pred is the kriging prediction 
    - s_pred is the kriging predictive variance.
    """
    return (z_obs - z_pred)/np.sqrt(ss_pred)

def logbias(y_true,y_pred):
    """ Log bias metric (dB) """
    return 10.0*np.log10(np.sum(y_true)/np.sum(y_pred))