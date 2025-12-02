import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np 
import shapely 
from pyproj import CRS

# =========================================================================================
# MISC.TOOLS FOR DATA LOADING, BINNING, MASKING
# =========================================================================================
def load_xarray_datarray(fn: str) -> xr.Dataset:
    """ Just a function to open a netcdf dataset """
    with xr.open_dataset(filename_or_obj=fn, engine='netcdf4',decode_times=False) as file:
        return file

def slice_in_equal_bins(series : pd.Series, bin_width: int) -> pd.Series :
    ''' This function bins a pandas Series according to the given bin width.
    It returns the binned Series.
    The values in this series are the bins label, which is defined as the lower bound of the bin range. 
    E.g : bin "0" spans from 0 to 0+width.
    Inputs :
        - series : pd.Series containing float values to bin
        - bin_width : integer of the desired bin width 
    Outputs : 
        - binned_series : pd.Series containing the bin label associated of each item. 
    '''
    bins = list(range(int(series.min()), int(series.max())+ bin_width + 1, bin_width))
    binned_series = pd.cut(series, bins, labels = bins[:-1])
    return binned_series

def bin_xrDataArray_time(da : xr.DataArray, res) -> xr.DataArray : 
    """ TODO """
    tmin, tmax = float(da.time.min()), float(da.time.max())
    bins = np.arange(tmin, tmax + int(res), int(res)) #type:ignore

    da_binned = da.groupby_bins('time', bins=bins).mean()
    da_binned = da_binned.rename({'time_bins': 'time'})
    return da_binned.assign_coords(time=bins[:-1])

def get_yrBP_from_itrace_time(days_after_start : int, start_year : int = 11700)->float:
    """ TODO"""
    return start_year + days_after_start / 365.0

def mask_country_shape(da : xr.DataArray | pd.DataFrame, 
                       country_names: list[str] = ["China"], 
                       all_touched : bool = True, 
                       shapefile : str = "../data/shapefiles/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp",
                       buffer_km : float = 0, 
                       ) -> xr.DataArray | pd.DataFrame :
    """ This functions defines a mask for the given country (polygone) and filters the points of the provided data array
    (which should have coords lon and lat) according to this mask. If all_touched is set to True (default), the mask 
    includes pixels that have at least one corner inside the country, otherwise it includes only pixels with their center
    inside the country.
    """
    # read country borders from shapefile
    world = gpd.read_file(shapefile)
    shape = world[world["NAME"].isin(country_names)].to_crs("EPSG:4326")
    geom = shape.union_all()
    
    # Apply buffer if requested
    if buffer_km > 0:
        # Project to equal-area or metric CRS (Asia-centered)
        shape_m = shape.to_crs(CRS.from_epsg(3857))  # Web Mercator, meters
        geom_m = shape_m.union_all()
        geom_m_buffered = geom_m.buffer(buffer_km * 1000)  # buffer in meters
        # Reproject back to lon/lat
        shape_buffered = gpd.GeoSeries([geom_m_buffered], crs=3857).to_crs(4326)
        geom = shape_buffered.union_all()

    if isinstance(da, xr.DataArray):
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
        raise TypeError("Input must be an xarray.DataArray or pandas.DataFrame")

def convert_lon_0_360_to_neg180_180(lon:np.ndarray) ->np.ndarray:
    """ Converts longuitudes in the interval O to 360 degreees to -180 to 180 degrees"""
    return (lon + 180) % 360 -180

def convert_lat_0_180_to_neg90_90(lat:np.ndarray) ->np.ndarray:
    """ Converts latitudes in the interval 0 to 180 degreees to -90 to 90 degrees"""
    return (lat + 90) % 180 -90

# =========================================================================================
# STATISTICAL METRICS
# =========================================================================================
def compute_r2(y_true : np.ndarray, y_pred : np.ndarray, weights = None) -> float:
    """ Computes the R^2 between true and predicted values 
    """
    if weights is None : 
        weights = np.ones_like(y_true)
    sst = np.nansum(weights * (y_true - np.nanmean(y_true))**2)
    ssr = np.nansum(weights * (y_true - y_pred)**2)
    r2 = 1 - (ssr / sst)
    return r2
