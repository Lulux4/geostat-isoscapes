
from pyproj import Transformer
import xarray as xr
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression
from pyproj import Geod
import gstools as gs
import skgstat as skg
from scipy.optimize import curve_fit
from tqdm import tqdm 
import pandas as pd
from . import utils,sisal_utils
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import Point
from shapely.ops import nearest_points
import geopandas as gpd

# ==========================================================
# Coordinates projections 
# ==========================================================

def get_projected_coords_and_vals(data_array : xr.DataArray, time : str | float, epsg : str = "EPSG:3857"):    
    ''' This function extracts a slice from a xarray DataArray at a given time,
    converts lat and lon coordinates to the chosen epsg projection, stack them to get an array of shape (n,2),
    and returns the stacked arrray as well as the flattened values associated to each point of the coordinates array.
    '''
    da_slice = data_array.sel(time=time)
    stacked = da_slice.stack(point=('lat','lon')).dropna('point')
    lons_s = stacked['lon'].values
    lats_s = stacked['lat'].values
    vals_s = stacked.values.astype(float)

    xs,ys = project_coords(lons_s,lats_s,epsg=epsg)
    coords_proj = np.column_stack([xs,ys])
    
    return coords_proj, vals_s

def project_coords(lons,lats,epsg="EPSG:3857"):
    ''' Project lon/lat coordinates to the given epsg projection'''
    transformer = Transformer.from_crs("EPSG:4326",epsg, always_xy=True)
    xs, ys = transformer.transform(lons, lats)
    return xs, ys 

# =========================================================
# Useful functions for detrending data
# =========================================================

def detrend_data_plane(coords, vals):
    ''' detrend data by fitting plane z ~ x + y via a linear regression and subtract it to vals.
    Returns the residuals
    '''
    lr = LinearRegression().fit(coords, vals)
    trend = lr.predict(coords)
    resid = vals - trend
    return resid

def detrend_multilinear(df_to_detrend, 
                        value_col="d18Op",
                        lat_col : str| None ="lat",
                        lon_col : str| None ="lon",
                        elev_col : str| None ="elevation",
                        dist_col : str| None ="dist_to_coast",
                        prect_col :str | None = 'prect',
                        include_lon=False,
                        include_lat=True,
                        include_elev=False,
                        include_dcoast=False,
                        include_prect=False):
    """
    Fit and remove a physically-based multilinear trend:
    trend = b0 + b_lat*lat + b_lat2*lat^2 + b_elev*elev + b_dist*dist + b_lon*lon
    Inputs :
        - df : pandas DataFrame. Must contain columns: value_col, lat_col, elev_col, dist_col (+ lon if include_lon=True)
        - include_lon : bool. Whether to include longitude term in the trend model (default False)
        - similar for all include_xxx
    Outputs:
        - df_out : DataFrame. Copy of df with added columns: 'trend', 'residual'
        - beta : model parameters
        - full_r2 : r2 of the trend model
    """
    df = df_to_detrend.copy()
    # Build design matrix
    X_cols = []
    if include_lat:
        df['lat2'] = df[lat_col] **2
        X_cols.append('lat')
        X_cols.append('lat2') # because i know that the latitude effect occurs as we go towards both poles.
    if include_lon:
        X_cols.append(lon_col)
    if include_elev:
        X_cols.append(elev_col)
    if include_dcoast:
        X_cols.append(dist_col)
    if include_prect:
        X_cols.append(prect_col)
    # Check that at least one predictor is included
    if len(X_cols)==0:
        raise ValueError('Must set at least one of the include_xxx parameters to True!')
    
    X = df[X_cols].values
    y = df[value_col].values
    # 
    # print('The predictors are ', X_cols)
    # Fit and get contributions
    contributions, beta = partial_r2_by_predictor(X, y,X_cols)

    # Compute fitted trend
    df['trend'] = beta[0] + X @ beta[1:] 

    # Detrend
    df['residual'] = y - df['trend'].values

    # Full R^2:
    full_r2 = utils.compute_r2(y,df['trend'].values)

    return df, full_r2, contributions, beta

def partial_r2_by_predictor(X, y,predictors):
    """
    Compute partial R^2 for each predictor in a linear model.
    X: 2D array (n, p) of predictors
    y: 1D array (n,)
    """

    # constant intercept
    X_full = np.column_stack([np.ones(len(X)), X])
    
    # full model
    beta_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
    resid_full = y - X_full @ beta_full
    ssr_full = np.sum(resid_full**2)
    
    # total variance
    sst = np.sum((y - y.mean())**2)

    p = X.shape[1]
    contributions = {}

    for j in range(p):
        # Remove predictor j
        X_reduced = np.column_stack([np.ones(len(X)), np.delete(X, j, axis=1)])
        beta_reduced = np.linalg.lstsq(X_reduced, y, rcond=None)[0]
        resid_reduced = y - X_reduced @ beta_reduced
        ssr_reduced = np.sum(resid_reduced**2)

        # Partial R square
        r2_j = (ssr_reduced - ssr_full)/sst
        contributions[predictors[j]] = r2_j

    return contributions, beta_full


# =========================================================
# Useful functions for variogram computation 
# =========================================================
def geodesic_condensed_distances(lonlat):
    """
    Compute condensed vector of all pairwise geodesic distances (in meters)
    between lon/lat points.
    """
    n = len(lonlat)
    geod = Geod(ellps="WGS84")

    # Upper-triangle indices for unique pairs
    i, j = np.triu_indices(n, k=1)
    lon1, lat1 = lonlat[i, 0], lonlat[i, 1]
    lon2, lat2 = lonlat[j, 0], lonlat[j, 1]
    # geod.inv returns (fwd_azimuth, back_azimuth, distance)
    _, _, dist = geod.inv(lon1, lat1, lon2, lat2)
    return dist

def variogram_gstools(coords, 
                      vals, 
                      n_bins : int, 
                      maxlag : float|None = None,
                      maxlag_factor : float = 0.65 
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ''' Computes variogram using gstools vario_estimate function. 
    Euclidean distances are used.
    '''
    # compute pairwise distances 
    dists = pdist(coords)
    
    # define max lag
    if maxlag is None:
        maxlag_ = dists.max() * maxlag_factor
    else :
        maxlag_ = maxlag

    # define bins **EDGES**
    bins_edges = np.linspace(0, maxlag_, n_bins + 1)
    
    # gstools variogram computation
    bin_centers, gamma, counts = gs.vario_estimate(coords.T, vals, bin_edges = bins_edges, return_counts = True)
    
    return bin_centers, gamma, counts


def variogram_with_gstat(df,
                         quantity:str,
                         sample_size=3000,
                         direction : int | None = None,
                         trend: str|None = None, 
                         nlags : int = 30, 
                         maxlag : float | str | None = 'median', 
                         centers : np.ndarray | None = None,
                         model : str ='spherical',
                         return_Variogram_object : bool = False,
                         seed : int = 42,
                         verbose : bool = True
                         )-> skg.Variogram | tuple[np.ndarray,np.ndarray, np.ndarray, object] :
    """Compute experimental variogram with sampling.
    TODO : write this fct doc 
    """
    # check if the month contains too much samples. If so, sample it down to the sample_size.
    if len(df) > sample_size:
        # print('downsampling')
        df = df.sample(sample_size, random_state=seed)
    
    vals = df[quantity].values
    if trend is not None: 
        if trend=='plane':
            # Remove trend by fitting a plane 
            XY = np.column_stack((df['x'], df['y']))
            vals = detrend_data_plane(XY,df[quantity].values)
        elif 'multilinear' in trend:
            # Remove trend by fitting a multilinear trend based on possibly many predictors : latitude (by default), elevation...
            df_detrended, r2, contributions, _ = detrend_multilinear(df,
                                                            value_col=quantity,
                                                            lat_col='lat',
                                                            lon_col='lon',
                                                            elev_col='elevation' if ('elevation' in df.columns)&('elev' in trend) else None,
                                                            dist_col='dist_coast_m' if ('dist_coast_m' in df.columns)&('dcoast' in trend) else None,
                                                            prect_col='prect' if ('prect' in df.columns)&('prect' in trend) else None,
                                                            include_lon='lon' in trend,
                                                            include_lat='lat' in trend,
                                                            include_elev='elev' in trend,
                                                            include_dcoast='dcoast' in trend,
                                                            include_prect='prect' in trend)
            vals = df_detrended['residual'].values
            if verbose :
                print(f'-> detrending with multilinear model, full R^2={r2:.3f}, contributions: {contributions}')
    # Compute the variogram 
    if direction is None :
        if centers is None :
            V = skg.Variogram(
                df[['x','y']].values,
                vals,
                n_lags=nlags,
                # normalize=True,
                maxlag=maxlag,
                model=model,
                use_nugget=True
            )
        else :
            V = skg.Variogram(
                df[['x','y']].values,
                vals,
                # normalize=True,
                bin_func = make_fixed_bin_func(centers), #type: ignore
                model=model,
                use_nugget=True
            )
    else :
        if centers is None :
            V = skg.DirectionalVariogram(
                    df[['x','y']].values,
                    vals,
                    n_lags=nlags,
                    maxlag=maxlag,
                    model=model,
                    use_nugget=True,
                    azimuth=direction, #type:ignore
                    tolerance = 22.5 )
        else : 
            V = skg.DirectionalVariogram(
                    df[['x','y']].values,
                    vals,
                    bin_func = make_fixed_bin_func(centers), #type:ignore
                    model=model,
                    use_nugget=True,
                    azimuth=direction, #type:ignore
                    tolerance = 22.5 )
        
    if return_Variogram_object:
        return V
    else :
        return V.bins, V.experimental, V.bin_count,V.fitted_model
    
def make_fixed_bin_func(centers):
    ''' returns a function with returning the (given) centers and associated edges, no matter the args provided.
    '''
    def fixed(*args, **kwargs):
        edges = np.concatenate(([centers[0] - (centers[1]-centers[0])/2],
                        (centers[:-1] + centers[1:])/2,
                        [centers[-1] + (centers[-1]-centers[-2])/2]))
        # print('edge shape :',edges.shape, ' centers shape :',centers.shape)
        return centers, edges
    return fixed

def get_weights_from_pair_counts(pair_counts: np.ndarray) -> np.ndarray :
    """ Computes the array of weigths associated to each bin from the pair count in the bin. 
    This is useful for computing weighted R^2 of variogram models, for instance.
    """
    mask_nans = np.isnan(pair_counts)
    weights = np.zeros_like(pair_counts,dtype=float)
    weights[~mask_nans] = np.sqrt(pair_counts[~mask_nans])
    return weights

def get_vario_parameters_dict(parameters):
    # retrieve the model parameters and give them the correct name:
    if len(parameters) == 3:
        range_, sill, nugget = parameters
        return {'range':range_,'sill':sill,'nugget':nugget}
    else:
        print('Parameters are ',parameters)
        raise NotImplementedError('TODO : implement parameters retrieval for this case')

# =========================================================================
# Iterative variogram computations
# =========================================================================

def iterative_variogram_computations(df,ref_bins = None,direction=None,trend='plane',verbose=False):
    """ TODO 
    """
    df['x'], df['y'] = project_coords(df['lon'].values, df['lat'].values, epsg="EPSG:3857")
    print('-> initializing loop...')
    bin_counts, gammas = [], []
    print('   starting loop...')
    for t, g in tqdm(df.groupby('time')):
        if g['d18Op'].notna().sum() < 30:
            continue
        try:
            if ref_bins is None :
                b, g_exp, bin_count,_ = variogram_with_gstat(g,
                                                             quantity='d18Op',
                                                             trend=trend,
                                                             maxlag='median',
                                                             direction=direction,
                                                             return_Variogram_object=False,
                                                             verbose=verbose) # type: ignore
                ref_bins = b
            else : 
                b, g_exp, bin_count,_ = variogram_with_gstat(g,
                                                             quantity='d18Op',
                                                             direction=direction,
                                                             trend=trend,
                                                             centers=ref_bins,
                                                             return_Variogram_object=False,
                                                             verbose = verbose) # type: ignore    
            bin_counts.append(bin_count)
            gammas.append(g_exp)
                
        except Exception as e:
            print(f"   Skipped {t}: {e}")
    print('finished loop') 
    return bin_counts, gammas, ref_bins

def aggregate_variograms(bin_counts,gammas,bins):
    """ TODO 
    """
    counts = np.vstack(bin_counts) # shape = n_time * n_bins
    total_counts = np.nansum(counts, axis=0) # shape n_bins
    weighted_mean_gamma = np.nansum(counts * np.vstack(gammas), axis=0) / total_counts # -> element-wise mutliplication of counts*gammas both of shape n_time * n_bins

    # remove empty bins if any
    mask_empty_bins = ~ (total_counts == 0) 
    weighted_mean_gamma_mask = weighted_mean_gamma[mask_empty_bins]
    bins_mask = bins[mask_empty_bins]
    total_counts_mask = total_counts[mask_empty_bins]
    df = pd.DataFrame({'lag':bins_mask, 'gamma':weighted_mean_gamma_mask, 'count':total_counts_mask})
    return df

# =========================================================================
# variogram **models** and fit function 
# =========================================================================
def spherical_model(h, sill, range_, nugget=0):
    """Spherical model for variograms 
    """
    gamma = np.where(
        h <= range_, # condition
        nugget + sill*(1.5*(h/range_) -0.5*(h/range_)**3), # if condition true
        nugget + sill # else
    )
    return gamma

def exponential_model(h, sill, range_, nugget=0):
    """ Exponential model for variograms
    """
    return nugget + sill*(1 - np.exp(-h/range_))

def gaussian_model(h, sill, range_, nugget=0):
    """Gaussian variogram model
    """
    return nugget + sill*(1 - np.exp(-(h**2)/(range_**2)))

def composite_model(h, *params):
    """
    Composite model supporting up to two components.
    Example parameter order:
      sill1, range1, sill2, range2, nugget
    """
    sill1, range1, sill2, range2, nugget = params
    return (spherical_model(h, sill1, range1, nugget=0) + gaussian_model(h, sill2, range2, nugget=0) + nugget )

def effective_range(bins, fitted_fct, frac=0.95):
    """Compute the lag where gamma(h) reaches the given fraction frac of total sill.
    """
    h = np.linspace(0, max(bins)*2, 500)
    gamma = fitted_fct(h)
    sill_total = np.nanmax(gamma)
    idx = np.where(gamma >= frac*sill_total)[0]
    return h[idx[0]] if len(idx) > 0 else np.nan

def fit_variogram_model(bins, gammas, model='spherical', initial_params=None, bounds=None, pair_counts=None,):
    """
    Fit a theoretical variogram model to empirical data (distances=bins and semivariances=gammas).

    Inputs :
        - bins : array
            Lag distances
        - gammas : array
            Semivariances
        - model : str
            'spherical', 'exponential', 'gaussian', or 'composite'
        - initial_params : list or None
            Starting guess for parameters
        - bounds : 2-tuple or None
            Lower and upper bounds for curve_fit
        - pair_counts : array or None.
            number of pairs in each bin, used for weighting the fit (not mandatory)

    Outputs : 
        - params : dict
            Optimal parameters
        - fitted_fct : callable
            The fitted model function
        - pcov : ndarray
            Covariance matrix of the fit
    """

    model_dict = {
        'spherical': (spherical_model, ['sill', 'range', 'nugget']),
        'exponential': (exponential_model, ['sill', 'range', 'nugget']),
        'gaussian': (gaussian_model, ['sill', 'range', 'nugget']),
        'composite': (composite_model, ['sill1', 'range1', 'sill2', 'range2', 'nugget'])
    }

    if model not in model_dict:
        raise ValueError(f"Unknown model '{model}'. Choose from {list(model_dict.keys())}.")

    func, param_names = model_dict[model]

    if initial_params is None:
        sill_guess = np.nanmax(gammas)
        range_guess = np.nanmax(bins) / 3
        nugget_guess = gammas[0]
        if model == 'composite':
            initial_params = [sill_guess/2, range_guess, sill_guess/2, range_guess*2, nugget_guess]
        else:
            initial_params = [sill_guess, range_guess, nugget_guess]

    if pair_counts is not None:
        pair_counts = np.asarray(pair_counts)
        # Weights proportional to sqrt(N) => sigma = 1 / sqrt(N) => curve_fit minimizes residual*sqrt(N)
        sigma = 1.0 / np.sqrt(np.maximum(pair_counts, 1))
    else:
        sigma = None 
    try : 
        popt, pcov = curve_fit(func, bins, gammas, p0=initial_params, bounds=bounds or (-np.inf, np.inf), sigma=sigma, absolute_sigma=False)
        fitted_fct = make_fitted_model_func(func,*popt)
        params = {name: val for name, val in zip(param_names, popt)}
        params['model_name'] = model
    except Exception as e :
        print(e)
        params, fitted_fct,pcov = None,None, None
    return params,fitted_fct,pcov

def make_fitted_model_func(f,*args,**kwargs):
    ''' Make a function that takes only lags in arguments and has fixed parameters '''
    def func(h):
        return f(h,*args,*kwargs)
    return func


# =============================================
# KRIGING
# =============================================
def get_sisal_data_for_kriging(res : int = 200, 
                               temp_ds_fn : str = '../data/temperature/temp_800ka_ann.nc',
                               countries : list = ['China'],
                               buffer_km : float = 500,
                               conversion : str = 'd18Op_VSMOW_exactconv'):
    """ Load SISAL data, sanatize it, convert it to drip water equivalents, sclice at the desired temporal resolution ."""
    
    data_df = sisal_utils.get_basic_cleaned_merged_sisal_data()

    # load temperature dataset
    temp_xda = utils.load_xarray_datarray(temp_ds_fn).temp

    # compute converted data
    data_df_drip_water = sisal_utils.retrieve_temperature_and_convert_speleothem_d18O(data_df,temp_xda=temp_xda,method='linear')
    # remove samples for which the conversion failed (usually due to T retrieval failure)
    data_df_drip_water = data_df_drip_water.dropna(subset=conversion)

    # Bin years to the desired temporal resolution
    chrono_series = data_df_drip_water['age'].copy()
    data_df_drip_water['binned_age'] = utils.slice_in_equal_bins(chrono_series,res)

    data_df_drip_water = data_df_drip_water.rename(columns={'latitude':'lat','longitude':'lon'})

    data_ready = utils.mask_country_shape(data_df_drip_water,buffer_km=buffer_km,country_names=countries)

    data_ready = data_ready.rename(columns={'lat':'latitude','lon':'longitude'}) # type:ignore

    return data_ready

def get_itracedata_for_external_drift(res=None,
                                      itrace_data_folder = '../data/modern/iTrace/',
                                      itrace_sim_prefix = 'b.e13.Bi1850C5.f19_g16.12ka.itrace.ice_ghg_orb_wtr.05.clm2.h0.',
                                      itrace_sim_suffix = '.800001-899912',
                                      include_snow = True,
                                      countries : list| None = None,
                                      buffer_km : float = 500,
                                      prect_da = False,
                                      delta18_da = False):
    """ TODO """
    # files to find and read :
    fn_itrace_RAIN_H218O = f'{itrace_data_folder}{itrace_sim_prefix}RAIN_H218O{itrace_sim_suffix}.nc'
    fn_itrace_RAIN_H2OTR = f'{itrace_data_folder}{itrace_sim_prefix}RAIN_H2OTR{itrace_sim_suffix}.nc'
    fn_itrace_RAIN = f'{itrace_data_folder}{itrace_sim_prefix}RAIN{itrace_sim_suffix}.nc'
    if include_snow :
        fn_itrace_SNOW_H218O = f'{itrace_data_folder}{itrace_sim_prefix}SNOW_H218O{itrace_sim_suffix}.nc'
        fn_itrace_SNOW_H2OTR = f'{itrace_data_folder}{itrace_sim_prefix}SNOW_H2OTR{itrace_sim_suffix}.nc'    
        fn_itrace_SNOW = f'{itrace_data_folder}{itrace_sim_prefix}SNOW{itrace_sim_suffix}.nc'

    # load files 
    file_rH218O = utils.load_xarray_datarray(fn_itrace_RAIN_H218O)
    file_rH2OTR = utils.load_xarray_datarray(fn_itrace_RAIN_H2OTR)
    if include_snow :
        file_sH218O = utils.load_xarray_datarray(fn_itrace_SNOW_H218O)
        file_sH2OTR = utils.load_xarray_datarray(fn_itrace_SNOW_H2OTR)
    
    # compute the delta18Op
    h218o = file_rH218O.RAIN_H218O
    h2o = file_rH2OTR.RAIN_H2OTR
    if include_snow :
        h218o += file_sH218O.SNOW_H218O
        h2o += file_sH2OTR.SNOW_H2OTR

    delta18 = ( h218o/h2o - 1.0) * 1000.0
    delta18 = delta18.where( (h2o> 1e-12) & (delta18 < 1e2) )  # avoid div by near-0 precip values and large positive outliers (delta18 should be mostly negative and small -20/+20)
    
    # Fix the temporal resolution 
    time_series = pd.Series(delta18.time.values)
    if res is None :
        maxres = time_series.sort_values().diff().max()
        print(f'-> regular temporal resolution of {maxres} days')
        res = maxres
    delta18_binned = utils.bin_xrDataArray_time(delta18,res=res)

    if countries is not None :
        delta18_binned = utils.mask_country_shape(delta18_binned,buffer_km=buffer_km,country_names=countries)  

    if prect_da :
        # load netcdf files 
        file_rain = utils.load_xarray_datarray(fn_itrace_RAIN)
        precip_da = file_rain.RAIN
        if include_snow :
            file_snow = utils.load_xarray_datarray(fn_itrace_SNOW)
            precip_da += file_snow.SNOW

        # bin time to match delta18_binned resolution
        binned_prect_da = utils.bin_xrDataArray_time(precip_da,res=res)
        binned_prect_da = binned_prect_da * res * 24 * 3600 # type:ignore
        if countries is not None :
            binned_prect_da = utils.mask_country_shape(binned_prect_da,buffer_km=buffer_km,country_names=countries) 
    # outputs differ depending on bools prect_da, delta18_da... Return df of xrdataarrays.
    if delta18_da : 
        if prect_da :
            return delta18_binned, binned_prect_da
        return delta18_binned
    
    data_df = delta18_binned.to_dataframe(name='d18Op').reset_index().dropna() # type:ignore
    data_df = data_df.rename(columns={'lon':'longitude','lat':'latitude'})
    return data_df

# =========================================================================
# External variables handling and interpolation
# ========================================================================

def add_external_variables_to_lonlat_df(df_orig : pd.DataFrame, 
                                        time : float,
                                        res = 31,
                                        variables : list[str]=['elev','dcoast','prect'],
                                        dem_file : str = "../data/elevation/ETOPO_2022_v1_60s_N90W180_surface.nc",
                                        coast_shapefile : str = "../data/shapefiles/ne_10m_coastline/ne_10m_coastline.shp",
                                        rain_file : str = '../data/modern/iTrace/b.e13.Bi1850C5.f19_g16.12ka.itrace.ice_ghg_orb_wtr.05.clm2.h0.RAIN.800001-899912.nc',
                                        snow_file : str = '../data/modern/iTrace/b.e13.Bi1850C5.f19_g16.12ka.itrace.ice_ghg_orb_wtr.05.clm2.h0.SNOW.800001-899912.nc',       
                                        ):
    print(f'-> Adding external variables {variables}.')
    df = df_orig.copy()
    # check that latitude and longitudes are defined symetrically around 0° 
    if any(df['lat'])>90:
        df['lat']=utils.convert_lat_0_180_to_neg90_90(np.array(df['lat'].values))
    if any(df['lon'])>180 :
        df['lon']=utils.convert_lon_0_360_to_neg180_180(np.array(df['lon'].values))
    
    # add columns with the variables specified in argument
    if 'elev' in variables :
        print(f'   > elevation data will be taken from file {dem_file}')
        df = interpolate_dem_at_latlon_points(df,dem_file = dem_file)
    if 'dcoast' in variables:
        print(f'   > coastlines will be taken from file {coast_shapefile}')
        df = compute_distance_to_coast(df,coast_shapefile)
    if 'prect' in variables :
        print(f'Prect rain will be taken from file {rain_file} \nPrect snow file will be taken from file {snow_file}')
        df = add_prect_to_itrace_df(df,time=time,res=res,rain_file=rain_file,snow_file=snow_file)
    return df

def interpolate_dem_at_latlon_points(df_latlon: pd.DataFrame, dem_file :str ="../data/elevation/ETOPO_2022_v1_60s_N90W180_surface.nc"):
    """ TODO 
    lat : -90 to 90
    lon : -180 to 180
    """
    df = df_latlon.copy()
    if any(df['lon']>180):
        df['lon']=utils.convert_lon_0_360_to_neg180_180(np.array(df['lon']))
    if any(df['lat']>90):
        df['lat']=utils.convert_lat_0_180_to_neg90_90(np.array(df['lat']))

    # Load global elevation grid (ETOPO for instance)
    ds = xr.open_dataset(dem_file,engine='netcdf4') # ds must have lat/lon coords and z variable
    lats_dem = ds['lat'].values
    lons_dem = ds['lon'].values
    elevation_grid = ds['z'].values

    # interpolation at df points
    interp_elev = RegularGridInterpolator(
        (lats_dem, lons_dem),
        elevation_grid,
        bounds_error=False,
        fill_value=np.nan
    )

    lats_df = df['lat'].values
    lons_df = df['lon'].values

    elev_at_obs = interp_elev(np.column_stack([lats_df, lons_df])) # type: ignore
    df['elevation'] = elev_at_obs

    # we seen there are points at -6000m (indonesia, in the ocean). This is a problem since I suppose that itrace simulation only models terrestrial points, but there was a buffer wround the coast that included some points in the ocean.
    # either i must manage to mask these points, or clip the values to sea level (or -423m, the lowest surface elevation?)
    df.loc[df['elevation']<0,'elevation']=0
    df = df.dropna(subset=['elevation'])
    
    return df

def compute_distance_to_coast(df_latlon, coast_shp_path="../data/shapefiles/ne_10m_coastline/ne_10m_coastline.shp"):
    """
    Add geodesic distance to nearest coastline (in meters) to a dataframe 
    containing columns 'lat' and 'lon'.
    Inputs:
        df : pandas.DataFrame, it must contain 'lat' and 'lon' columns.
        coast_shp_path : str, the path to a coastline shapefile.
    Outputs:
        df : pandas.DataFrame, same dataframe as input but with an extra column 'dist_coast_m'.
    """
    df = df_latlon.copy()

    if any(df['lon']>180):
        df['lon'] = utils.convert_lon_0_360_to_neg180_180(df['lon'].values)
    if any(df['lat']>90):
        df['lat'] = utils.convert_lat_0_180_to_neg90_90(df['lat'].values)
    
    coast = gpd.read_file(coast_shp_path).to_crs("EPSG:4326")

    geod = Geod(ellps="WGS84")

    distances = []
    for _, row in df.iterrows():
        pt = Point(row["lon"], row["lat"])
        # find actual nearest geometry, not just BB neighbors
        nearest_idx = coast.sindex.nearest(pt)[1]
        nearest_geom = coast.geometry.iloc[nearest_idx]

        # compute geodesic distance
        p1, p2 = nearest_points(pt, nearest_geom)
        _, _, dist = geod.inv(p1.iloc[0].x, p1.iloc[0].y, p2.iloc[0].x, p2.iloc[0].y) # type: ignore
        distances.append(dist)

    df["dist_coast_m"] = distances
    return df

def add_prect_to_itrace_df(df,
                           time,
                           res,
                           rain_file : str = '../data/modern/iTrace/b.e13.Bi1850C5.f19_g16.12ka.itrace.ice_ghg_orb_wtr.05.clm2.h0.RAIN.800001-899912.nc',
                           snow_file :str = '../data/modern/iTrace/b.e13.Bi1850C5.f19_g16.12ka.itrace.ice_ghg_orb_wtr.05.clm2.h0.SNOW.800001-899912.nc' 
                           ):
    """ TODO
    Prect = total atmospheric precipitation that reached the ground, including snowfall and rainfall, in water mm equivalents.
    /!/ For consistency, rain and snow files must be the outputs of the same piece of simulation as the data in df : same grid, same time step.
    inputs :
        rain file must be a netcdf dataset containing variable RAIN and dimensions lat,lon,time
        snow file must be a netcdf dataset containing variable SNOW and dimensions lat,lon,time
    output: 
        df
    """
    # load netcdf files 
    file_rain = utils.load_xarray_datarray(rain_file)
    file_snow = utils.load_xarray_datarray(snow_file)
    # file_rain.info()
    # file_snow.info()
    rain_da = file_rain.RAIN
    snow_da = file_snow.SNOW
    
    # compute total precip 
    prect_da = rain_da + snow_da
    
    # bin time to match itrace_da resolution
    binned_prect_da = utils.bin_xrDataArray_time(prect_da,res=res)
    binned_prect_da = binned_prect_da * res * 24 * 3600 # convert the flux (precipitation rate) to the "res" days total (typicaly res=31 -> monthly amount) 
    
    df_prect=binned_prect_da.sel(time=time).to_dataframe(name='prect').reset_index().dropna(subset='prect') # type:ignore
    
    df_prect['lon'] = utils.convert_lon_0_360_to_neg180_180(df_prect['lon'])

    df = pd.merge(df,df_prect,how='left',on=['lon','lat','time'])
    
    return df