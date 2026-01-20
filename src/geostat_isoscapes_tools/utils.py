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

############################################
# ROOT DIR RETRIEVAL
############################################
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

def slice_in_equal_bins(series : pd.Series, bin_width: int) -> pd.Series :
    ''' This function bins a pandas Series according to the given bin width.
    It returns the binned Series.
    The values in this series are the bins label, which is defined as the lower bound of the bin range. 
    E.g : bin "0" spans [0,0+width[.
    Inputs :
        - series : pd.Series containing float values to bin
        - bin_width : integer of the desired bin width 
    Outputs : 
        - binned_series : pd.Series containing the bin label associated of each item. 
    '''
    bins = list(range(int(series.min()), int(series.max())+ bin_width + 1, bin_width))
    binned_series = pd.cut(series, bins, labels = bins[:-1],right=False)
    return binned_series

def bin_xrDataArray_time(da : xr.DataArray, res) -> xr.DataArray : 
    """ TODO """
    tmin, tmax = float(da.time.min()), float(da.time.max())
    bins = np.arange(tmin, tmax + int(res), int(res)) #type:ignore

    da_binned = da.groupby_bins('time', bins=bins).median()
    da_binned = da_binned.rename({'time_bins': 'time'})
    return da_binned.assign_coords(time=bins[:-1])

def get_yrBP_from_itrace_time(months_after_start : int | pd.Series, start_year : float = 12001)->float:
    """ Retrieve the year (+decimals) depending on a start year in yr BP (positive number) and the number of months spent since this start year.
    """
    return -( -start_year + months_after_start/12) # type:ignore

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
    sst = np.nansum(weights * (y_true - np.nanmean(y_true))**2)
    ssr = np.nansum(weights * (y_true - y_pred)**2)
    r2 = 1 - (ssr / sst)
    return r2

def r2_adj(y_true : np.ndarray, y_pred : np.ndarray, p : int, weights = None):
    """ Compute the r2 adjusted for the number of parameters in the prediction model """
    if weights is None : 
        weights = np.ones_like(y_true)
    n = len(y_true)
    sst = np.nansum(weights * (y_true - np.nanmean(y_true))**2)
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
