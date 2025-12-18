
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
from . import utils,sisal_utils,variogram_models
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import Point
from shapely.ops import nearest_points
import geopandas as gpd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import json 

# ==========================================================
# Coordinates projections 
# ==========================================================

def get_projected_coords_and_vals(data_array : xr.DataArray, time : str | float, epsg : str = "EPSG:3857",lat='lat',lon='lon'):    
    ''' This function extracts a slice from a xarray DataArray at a given time,
    converts lat and lon coordinates to the chosen epsg projection, stack them to get an array of shape (n,2),
    and returns the stacked arrray as well as the flattened values associated to each point of the coordinates array.
    '''
    da_slice = data_array.sel(time=time)
    stacked = da_slice.stack(point=(lat,lon)).dropna('point')
    lons_s = stacked[lon].values
    lats_s = stacked[lat].values
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
    Returns the residuals and the fitted plane
    '''
    lr = LinearRegression().fit(coords, vals)
    trend = lr.predict(coords)
    resid = vals - trend
    return resid, lr.get_params()

def detrend_multiple_linear_regression(df_to_detrend, 
                                    value_col="d18Op",
                                    features: str = 'lat',
                                    lat_col : str| None = "lat",
                                    lon_col : str| None = "lon",
                                    ele_col : str| None = "ele",
                                    D_col : str| None = "D",
                                    P_col :str | None = 'P',
                                    ):
    """
    Fit and remove a physically-based multiple linear trend:
    trend = X@a = b0 + b_lat*lat + b_lat2*lat^2 + b_elev*elev + b_dist*dist + b_lon*lon
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
    if 'lat' in features:
        if 'lat_' in features : 
            X_cols.append(lat_col)
        if 'latabs' in features :
            df['|lat|'] = np.abs(df[lat_col])
            X_cols.append('|lat|')
        if 'latsign' in features : 
            df['latsign'] = np.sign(df[lat_col])
            X_cols.append('latsign')
        if 'latquad' in features :
            df['lat2'] = df[lat_col]**2
            X_cols.append('lat2')
            if not (lat_col in X_cols):
                X_cols.append(lat_col)
        if 'latsqrt' in features :
            df['sqrt|lat|'] = np.sqrt(np.abs(df[lat_col]))
            X_cols.append('sqrt|lat|')
        if 'latReLU' in features : 
            df['latReLU'] = np.maximum(0,df[lat_col]) # ReLU -> to create asymetric PW linear activation
            X_cols.append('latReLU')
    if 'lon' in features:
        X_cols.append(lon_col)
    if 'ele' in features:
        X_cols.append(ele_col)
    if 'D' in features:
        X_cols.append(D_col)
    if 'P' in features:
        X_cols.append(P_col)

    # Check that at least one predictor is included
    if len(X_cols)==0:
        raise ValueError('Must set at least one of the include_xxx parameters to True!')
    X = df[X_cols].values
    y = df[value_col].values
    
    result_dict = fit_multiple_linear_model(X, y,X_cols)

    # beta = result_dict['parameters']
    # df['trend'] = beta[0] + X @ beta[1:] 
    df['trend'] = result_dict['y_pred']
    df['residual'] = y - df['trend'].values

    return df, result_dict

def fit_multiple_linear_model(X, y,predictors):
    """
    Fits a multiple linear model.
    Computes partial R^2 for each predictor in the multiple linear model.
    X: 2D array (n, p) of predictors
    y: 1D array (n,)
    """
    (n,p) = X.shape

    # add constant intercept
    # X_full = np.column_stack([np.ones(len(X)), X])
    X_full = sm.add_constant(X)

    # model with all predictors
    model_full  = sm.OLS(y, X_full).fit()
    beta_full =  model_full.params
    ssr_full = np.sum(model_full.resid**2)
    
    # r2 and adjusted r2
    y_pred = model_full.fittedvalues
    y_res = y-y_pred
    mae = np.mean(np.abs(y - y_pred))
    sst = np.sum((y - y.mean())**2)
    full_r2 = 1- ssr_full/sst
    adj_r2 = 1 - (ssr_full/(n-p-1)) / (sst/(n-1))

    partial_r2 = {}
    for j in range(X.shape[1]):
        # Remove predictor j
        X_reduced = np.delete(X_full, j+1, axis=1)
        model_reduced = sm.OLS(y, X_reduced).fit()
        ssr_reduced = np.sum(model_reduced.resid**2)

        # Partial R square
        r2_j = (ssr_reduced - ssr_full)/sst
        partial_r2[predictors[j]] = r2_j
    
    # p-values and std
    p_values = dict(zip(["intercept"] + predictors, model_full.pvalues))
    coefficient_std = dict(zip(["intercept"] + predictors, model_full.bse))
       
    # dict of results 
    result_dict = {'parameters':dict(zip(["intercept"] + predictors,beta_full)),
                   'y_pred':list(np.array(y_pred,dtype=float)),
                   'y_res': list(np.array(y_res,dtype=float)),
                   'y' : list(np.array(y,dtype=float)),
                   'mae':float(mae),
                   'parameters_std': coefficient_std,
                   'r2':float(full_r2),
                   'adj_r2': float(adj_r2),
                   'partial r2': partial_r2,
                   'p':p_values,
                   }
    # for the intercept, it makes no sense to compute vif or partial r2 
    if len(predictors)>1:
        vif = {predictors[i]: variance_inflation_factor(X_full[:, 1:], i) for i in range(len(predictors)) } # bc we need to exclude the intercept
        result_dict['vif']=vif
        result_dict["vif"]["intercept"] = None
    result_dict["partial r2"]["intercept"] = None
    
    return result_dict

def multiple_linear_result_dict_to_df(result_dict):
    """ TODO
    for printing results (df is printed more nicely than a dict)
    """
    res_df = pd.DataFrame({
                    "name": result_dict['parameters_std'].keys(),
                    "beta": result_dict["parameters"].values(),
                    "std_error": result_dict["parameters_std"].values(),
                    "p_value": result_dict["p"].values(),
                })
                
    if len(result_dict['parameters'])>2:
        vif = pd.DataFrame.from_dict(result_dict['vif'].items()).rename(columns={0:'name',1:'vif'})
        pr2 = pd.DataFrame.from_dict(result_dict["partial r2"].items()).rename(columns={0:'name',1:'partial r2'})
        res_df = pd.merge(res_df,vif,on='name').merge(pr2,on='name')
    
    return res_df

def trend_removal(trend: str | None,df : pd.DataFrame,quantity : str, verbose: bool = False,lat='lat',lon='lon'):
    """ Fits and remove trend model according to the name of the model passed in argument. Also returns the results (coeffs and metrics) 
    of the fit. """
    if trend is None :
        if verbose : print('Arg trend=None : returning raw values and empty dict as result')
        return df[quantity],None
    
    elif trend=='plane':
        # Remove trend by fitting a plane 
        XY = np.column_stack((df['x'], df['y']))
        vals, results = detrend_data_plane(XY,df[quantity].values)
    
    elif 'multiple_linear' in trend:
        # Remove trend by fitting a multilinear trend based on possibly many predictors : latitude (by default), elevation...
        if verbose : print(f'trend to model : {trend}')
        df_detrended, results = detrend_multiple_linear_regression(df,
                                                                value_col=quantity,
                                                                lat_col = lat,
                                                                lon_col = lon,
                                                                ele_col = 'ele' if ('ele' in df.columns)&('ele' in trend) else None,
                                                                D_col = 'D' if ('D' in df.columns)&('D' in trend) else None,
                                                                P_col = 'P' if ('P' in df.columns)&('P' in trend) else None,
                                                                features = trend
                                                                )
        vals = df_detrended['residual'].values
        if verbose :
            res_df = multiple_linear_result_dict_to_df(result_dict=results)
            print(f"-> detrending with multiple linear regression, R2 is {results['r2']:.3f} detailed metrics are :\n",res_df)
    else : 
        raise ValueError(f'Trend model {trend} is not supported.')
    
    return vals,results

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


def variogram_with_gstat(df : pd.DataFrame,
                         quantity : str = 'd18Op',
                         sample_size : int =3000,
                         direction : float | None = None,
                         trend: str|None = None, 
                         nlags : int = 30, 
                         maxlag : float | str | None = 'median', 
                         centers : np.ndarray | None = None,
                         model : str ='spherical',
                         return_Variogram_object : bool = False,
                         seed : int = 42,
                         verbose : bool = True,
                         tolerance = 22.5,
                         x= 'x',
                         y= 'y'
                         ) :
    """Compute experimental variogram with sampling.
    TODO : write this fct doc 
    """
    # check if the month contains too much samples. If so, sample it down to the sample_size.
    if len(df) > sample_size:
        # print('downsampling')
        df = df.sample(sample_size, random_state=seed)
    
    vals = df[quantity].values

    trend_results = None
    if trend is not None: 
        vals, trend_results = trend_removal(trend,df,quantity,verbose)
    
    # Compute the variogram
    # =================
    # UNDIRECTIONAL
    # ================= 
    if direction is None :
        # ==============
        # maxlag & nlags
        # ==============
        if centers is None:
            V = skg.Variogram(
                df[[x,y]].values,
                vals,
                n_lags=nlags,
                # normalize=True,
                maxlag=maxlag,
                model=model,
                use_nugget=True
            )
        # ==============
        # fixed bins
        # ==============
        else :
            V = skg.Variogram(
                df[[x,y]].values,
                vals,
                # normalize=True,
                bin_func = make_fixed_bin_func(centers), #type: ignore
                model=model,
                use_nugget=True
            )
    # =================
    # DIRECTIONAL
    # =================
    else :
        # ==============
        # maxlag & nlags
        # ==============
        if centers is None :
            V = skg.DirectionalVariogram(
                    df[[x,y]].values,
                    vals,
                    n_lags=nlags,
                    maxlag=maxlag,
                    model=model,
                    use_nugget=True,
                    azimuth=direction, #type:ignore
                    tolerance = tolerance )
        # ==============
        # fixed bins
        # ==============
        else : 
            V = skg.DirectionalVariogram(
                    df[[x,y]].values,
                    vals,
                    bin_func = make_fixed_bin_func(centers), #type:ignore
                    model=model,
                    use_nugget=True,
                    azimuth=direction, #type:ignore
                    tolerance = tolerance )
        
    if return_Variogram_object:
        return V, trend_results
    else :
        return V.bins, V.experimental, V.bin_count,V.fitted_model, trend_results
    
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

# ========= deprecated : iterative vario looping on df grouped by time. See below for new version 
# def iterative_variogram_computations_df(df,ref_bins = None,direction=None,trend='plane',verbose=False):
#     """ TODO 
#     """
#     df['x'], df['y'] = project_coords(df['lon'].values, df['lat'].values, epsg="EPSG:3857")
#     print('-> initializing loop...')
#     bin_counts, gammas = [], []
#     print('   starting loop...')
#     for t, g in tqdm(df.groupby('time')):
#         if g['d18Op'].notna().sum() < 30:
#             continue
#         try:
#             if ref_bins is None :
#                 b, g_exp, bin_count,_ = variogram_with_gstat(g,
#                                                              quantity='d18Op',
#                                                              trend=trend,
#                                                              maxlag='median',
#                                                              direction=direction,
#                                                              return_Variogram_object=False,
#                                                              verbose=verbose) # type: ignore
#                 ref_bins = b
#             else : 
#                 b, g_exp, bin_count,_ = variogram_with_gstat(g,
#                                                              quantity='d18Op',
#                                                              direction=direction,
#                                                              trend=trend,
#                                                              centers=ref_bins,
#                                                              return_Variogram_object=False,
#                                                              verbose = verbose) # type: ignore    
#             bin_counts.append(bin_count)
#             gammas.append(g_exp)
                
#         except Exception as e:
#             print(f"   Skipped {t}: {e}")
#     print('finished loop') 
#     return bin_counts, gammas, ref_bins

def iterative_variogram_computations(ds : xr.DataArray | xr.Dataset,
                                    quantity : str ="d18Op",
                                    ref_bins : np.ndarray | None =None,
                                    direction : float | None = None,
                                    tolerance : float = 22.5,
                                    trend : str| None = None,
                                    maxlag : float | None | str = None,
                                    nlags : int = 20,
                                    mask : pd.DataFrame | None = None,
                                    trend_before_masking : bool = True,
                                    lat: str = 'lat',
                                    lon : str ='lon',
                                    verbose : bool =False):
    """
    Vario computation looping over an xarray dataset.

    Inputs: 
        ds : xr.Dataset containing lon, lat, time, and the variable quantity
        quantity : str, name of the column containing the data to study
        ref_bins : array-like of float, fixed bins
        direction : float, azimuth of the directional variogram in degrees (0=East of the coord plane)
        trend : str, name of the trend model to fit and remove.
        maxlag : str or float, max lag of the vario
        nlags : int, number of lags of the vario
        mask : xr datarray with lat lon and time dims, used to mask some locations or times.
        trend_before_masking : bool, specifies if trend removal should be applied before masking (True) or after (False).
        verbose: bool, sets the verbosity of the function
    """
    quantity_tmp = quantity
    trend_tmp = trend
    xs, ys, lons, lats, exog_df,trend_results = None,None,None,None,None,{}
    bin_counts, gammas = [], []

    for v in [lon, lat, quantity]:
        if v not in ds:
            raise ValueError(f"Dataset must contain a '{v}' variable.")

    for t, ds_t in tqdm(ds.groupby("time")):
        # Convert to DataFrame
        df = ds_t.to_dataframe().reset_index()
        
        # Convert lat lon ranges :
        if any(df[lon]>180) & (lons is None):
            lons = utils.convert_lon_0_360_to_neg180_180(np.array(df[lon]))
        if any(df[lat]>90) & (lats is None):
            lats= utils.convert_lat_0_180_to_neg90_90(np.array(df[lat]))
        if any(df[lon]>180): df[lon]=lons
        if any(df[lat]>90): df[lat]=lats

        # Skip if not enough data
        if df[quantity].notna().sum() < 30:
            continue

        # Compute projected coordinates
        if (xs is None)&(ys is None):
            xs,ys = project_coords(df[lon].values,df[lat].values,epsg="EPSG:3857")
        df["x"]=xs
        df["y"]=ys

        # Add exog variables
        if (trend is not None):
            if (exog_df is None) :
                if 'multiple_linear' in str(trend) :
                    variables = trend.split('_') #type:ignore
                    variables = [v for v in variables if v in ['ele','D']]
                    cols_exog = np.unique(np.array(variables)).tolist()
                    cols_exog.extend([lat,lon])
                    exog_df = add_external_variables_to_lonlat_df(df,variables=variables,lat=lat,lon=lon,verbose=False)[cols_exog]
            df = df.merge(exog_df,on=[lat,lon]) # type:ignore
            df = df.dropna(axis=0) # removes locs where exog data was not available
        
        # If specified : fit and remove trend here instead of inside variogram_with_gstat
        if (trend is not None) & (mask is not None) and (trend_before_masking) :
            if verbose : print(f'   removing trend {trend} using all available locations.')
            df['resid'],dict_trend = trend_removal(trend,df,quantity,verbose=verbose,lat=lat,lon=lon)
            trend_results[str(t)] = dict_trend
            quantity = 'resid'
            trend = None
        
        # If specified : mask data with provided mask (eg mask itrace to keep only sisal pts neighborhoods)
        if mask is not None :
            # sanity check for the latlon format
            if any(mask[lon]>180):
                mask[lon] = utils.convert_lon_0_360_to_neg180_180(np.array(mask[lon]))
            if any(mask[lat]>90):
                mask[lat] = utils.convert_lat_0_180_to_neg90_90(np.array(mask[lat]))
            if verbose : print('   masking some locations with the mask that was provided.')
            df = utils.apply_spatial_mask(df,mask,lat,lon,'mask')
                
        try:
            if verbose : print(f'   computing semivariances of the {quantity}')
            if ref_bins is None:
                b, g_exp, bin_count, _,_ = variogram_with_gstat(df,                  # type: ignore
                                                            quantity=quantity,
                                                            trend=trend,
                                                            maxlag=maxlag,
                                                            nlags=nlags,
                                                            direction=direction,
                                                            tolerance = tolerance,
                                                            return_Variogram_object=False,
                                                            verbose=verbose,
                                                            x='x',
                                                            y='y'
                                                        )
                ref_bins = np.array(b)
            else:
                b, g_exp, bin_count, _,_ = variogram_with_gstat(df,                  # type:ignore
                                                            quantity=quantity,
                                                            direction=direction,
                                                            tolerance = tolerance,
                                                            trend=trend,
                                                            centers=ref_bins,        # type:ignore
                                                            return_Variogram_object=False,
                                                            verbose=verbose,
                                                            x='x',
                                                            y='y'
                                                        )

            bin_counts.append(bin_count)
            gammas.append(g_exp)
            quantity = quantity_tmp
            trend  = trend_tmp
        except Exception as e:
            raise e

    return bin_counts, gammas, ref_bins, trend_results

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

def iterate_and_aggregate_variograms(data_ds : xr.DataArray | xr.Dataset,
                                     fp : str,
                                     config_dict : dict, 
                                     mask_pts : pd.DataFrame | None = None,
                                     data_cols : dict = {'lat':'lat','lon':'lon','quantity':'d18Op'},
                                     verbose : bool = False):
    """ TODO """
    # if a mask must be applied, define it here 
    mask_df = None
    if mask_pts is not None :
        if verbose : print('Defining spatial mask around anchor points')
        _, mask_da = utils.mask_union_of_circles_around_pts(data_to_mask=data_ds,
                                                            df_ref_pts=mask_pts,
                                                            radius_km=config_dict['mask radius [km]'],
                                                            lat_name=data_cols['lat'],
                                                            lon_name=data_cols['lon'],
                                                            verbose = verbose) # type:ignore
        mask_df = mask_da.to_dataframe(name='mask').reset_index().dropna()[['lon','lat','mask']].drop_duplicates() #type:ignore
        
    # Loop over the different time slices to compute the each variogram
    if verbose : print('Start variogram computation iterations')
    bin_count,gammas, ref_bins, results_dict = iterative_variogram_computations(data_ds,
                                                                                quantity = data_cols['quantity'],
                                                                                trend = config_dict['trend'],
                                                                                maxlag = config_dict['maxlag'],
                                                                                nlags = config_dict['nlags'],
                                                                                mask = mask_df,
                                                                                ref_bins = config_dict['centers'],
                                                                                direction = config_dict['direction'],
                                                                                tolerance = config_dict['tolerance'],
                                                                                trend_before_masking = config_dict['trend_before_mask'],
                                                                                lat = data_cols['lat'],
                                                                                lon = data_cols['lon'],
                                                                                verbose=verbose)
    # aggregate semivariances
    if verbose : print('Aggregate variograms')
    df_all = aggregate_variograms(bin_counts=bin_count,gammas=gammas,bins=ref_bins)
    # save results
    # if config_dict['direction'] is not None : # add info on direction in file names
    #     fp+=f"d{str(config_dict['direction'])}"
    
    df_all.to_csv(f'{fp}vario_{config_dict["direction"]}_df.csv')
    with open(f'{fp}trend_metrics{config_dict["direction"]}.json', 'w') as f:
    #     for key,value in results_dict.items():
    #         print(key)
    #         print(type(value))
    #         if type(value)==dict:
    #             for k,v in value.items():
    #                 print('  ',k)
    #                 print('  ',type(v))
    #                 if type(v)==list:
    #                     print('      ',type(v[0]))
    #         print('----')
        json.dump(results_dict, f)
    print(f'outputs saved in folder {fp}, files vario_df.csv and trend_metrics.json')

# =========================================================================
# Vario fit function 
# =========================================================================
def effective_range(bins, fitted_fct, frac=0.95):
    """Compute the lag where gamma(h) reaches the given fraction frac of total sill.
    """
    h = np.linspace(0, max(bins)*2, 500)
    gamma = fitted_fct(h)
    sill_total = np.nanmax(gamma)
    idx = np.where(gamma >= frac*sill_total)[0]
    return h[idx[0]] if len(idx) > 0 else np.nan

def fit_variogram_model(bins, gammas, model_name='spherical', initial_params=None, bounds=None, pair_counts=None,):
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

    # model_dict = {
    #     'spherical': (spherical_model, ['sill', 'range', 'nugget']),
    #     'exponential': (exponential_model, ['sill', 'range', 'nugget']),
    #     'gaussian': (gaussian_model, ['sill', 'range', 'nugget']),
    #     'composite': (composite_model, ['sill1', 'range1', 'sill2', 'range2', 'nugget'])
    # }

    # func, param_names = model_dict[model]
    model = variogram_models.define_model(model_name)
    func = model.get_model_func()
    param_names = model.params

    if initial_params is None:
        sill_guess = np.nanmax(gammas)
        range_guess = np.nanmax(bins) / 3
        nugget_guess = gammas[0]
        if '+' in model_name:
            initial_params = [nugget_guess,range_guess, sill_guess/2,range_guess*2, sill_guess/2]
        else:
            initial_params = [range_guess,sill_guess, nugget_guess]

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
        params['model_name'] = model_name
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
                               temp_ds_fn : str = '/data/temperature/temp_800ka_ann.nc',
                               countries : list | None = ['China'],
                               buffer_km : float = 500,
                               conversion : str = 'd18Op_VSMOW_exactconv',
                               verbose : bool = True):
    """ Load SISAL data, sanatize it, convert it to drip water equivalents, sclice at the desired temporal resolution (in years!) .
    """
    print('loading sisal data')
    data_df = sisal_utils.get_basic_cleaned_merged_sisal_data(verbose=verbose)

    # load temperature dataset
    temp_xda = utils.load_xarray_datarray(utils.get_project_root()+temp_ds_fn).temp

    # compute converted data
    data_df_drip_water = sisal_utils.retrieve_temperature_and_convert_speleothem_d18O(data_df,temp_xda=temp_xda,method='linear',verbose=verbose)
    # remove samples for which the conversion failed (usually due to T retrieval failure)
    data_df_drip_water = data_df_drip_water.dropna(subset=conversion)

    # Bin years to the desired temporal resolution
    chrono_series = data_df_drip_water['age'].copy()
    data_df_drip_water['binned_age'] = utils.slice_in_equal_bins(chrono_series,res)
    
    data_df_drip_water = data_df_drip_water.rename(columns={'latitude':'lat','longitude':'lon'})
    if countries is not None :
        data_df_drip_water = utils.mask_country_shape(data_df_drip_water,buffer_km=buffer_km,country_names=countries)

    # data_ready = data_df_drip_water.rename(columns={'lat':'latitude','lon':'longitude'}) # type:ignore
    print('sisal dataframe is ready')
    return data_df_drip_water

def get_preprocessed_itrace_data(res=None,
                            data_folder = '/data/modern/iTrace/',
                            sim_prefix = 'b.e13.Bi1850C5.f19_g16',
                            sim_kyr = 12,
                            sim_forcings = 'ice_ghg_orb_wtr',
                            sim_num = '05',
                            sim_model = 'clm2.h0',
                            sim_suffix = '800001-899912',
                            include_snow = True,
                            countries : list| None = None,
                            buffer_km : float = 500,
                            P : bool = False,
                            format : str = 'df',
                            verbose : bool = True) :
    """ TODO 
    res : in ***MONTHS***
    format : df or xr
    P = precip quantity
    """
    print('loading itrace files')
    # files to find and read :
    rootdir = utils.get_project_root()
    fn_merged = rootdir+f'{data_folder}{sim_prefix}.{sim_kyr}ka.itrace.{sim_forcings}.{sim_num}.{sim_model}'
    fn_itrace_RAIN_H218O = f'{fn_merged}.RAIN_H218O.{sim_suffix}.nc'
    fn_itrace_RAIN_H2OTR = f'{fn_merged}.RAIN_H2OTR.{sim_suffix}.nc'
    fn_itrace_RAIN = f'{fn_merged}.RAIN.{sim_suffix}.nc'
    if include_snow :
        fn_itrace_SNOW_H218O = f'{fn_merged}.SNOW_H218O.{sim_suffix}.nc'
        fn_itrace_SNOW_H2OTR = f'{fn_merged}.SNOW_H2OTR.{sim_suffix}.nc'    
        fn_itrace_SNOW = f'{fn_merged}.SNOW.{sim_suffix}.nc'

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
    
    # The itrace doc specifies explicetely that the temporal resolution is in months, so we can define the time dimension in terms of months after a start year.
    if verbose : print('This dataset contains ',len(delta18.time), ' time steps.') # just to check
    timearray = range(0,len(delta18.time),1)
    delta18 = delta18.assign_coords(time=('time',timearray))
    delta18 = delta18.assign_coords(time = delta18.time.assign_attrs(units=f"months since start year ({sim_kyr} ka)"))
    
    if res is not None :
        if verbose : print(f'   bins of width={res} days ({res//365} years)') # type:ignore
        delta18 = utils.bin_xrDataArray_time(delta18,res=res)

    if countries is not None :
        delta18 = utils.mask_country_shape(delta18,buffer_km=buffer_km,country_names=countries)  

    if P :
        if verbose : print('  Loading files for total precipitation info')
        # load netcdf files 
        file_rain = utils.load_xarray_datarray(fn_itrace_RAIN)
        precip_da = file_rain.RAIN
        if verbose : print(f'   rain file info : {precip_da.attrs}')
        if include_snow :
            file_snow = utils.load_xarray_datarray(fn_itrace_SNOW)
            precip_da += file_snow.SNOW
        
        # set same time unit as delta18
        precip_da = precip_da.assign_coords(time =('time',timearray))
        precip_da = precip_da.assign_coords(time = precip_da.time.assign_attrs(units=f"months since start year ({sim_kyr} ka)"))
        precip_da = precip_da * 32 * 24 * 3600 # integrate over bin width (natural binwidth is 31 days)

        if res is not None :
            # bin time to match the delta18 resolution
            precip_da = utils.bin_xrDataArray_time(precip_da,res=res)
            precip_da = precip_da * res # uniform integration over the bin width

        if countries is not None :
            precip_da = utils.mask_country_shape(precip_da,buffer_km=buffer_km,country_names=countries) 
        d18_P_ds = xr.Dataset({'d18Op':delta18, 'P': precip_da})
    
    # outputs differ depending on bools P_da, delta18_da... Return df of xrdataarrays.
    if (not P)&(format=='xr') : return delta18
    if P&(format=='xr') : return d18_P_ds
    
    if verbose : print('   converting xr to DataFrame (~2 minutes)')
    if format=='df' :
        if P : 
            d18_P_df= d18_P_ds.to_dataframe().reset_index().dropna(subset=['P','d18Op'])
            print('done')
            return d18_P_df
        else :
            d18_df = delta18.to_dataframe(name='d18Op').reset_index().dropna() #type:ignore
            print('done')
            return d18_df

# =========================================================================
# External variables handling and interpolation
# ========================================================================

def add_external_variables_to_lonlat_df(df_orig : pd.DataFrame, 
                                        variables : list[str]=['ele','D'],
                                        lat = 'lat',
                                        lon = 'lon',
                                        dem_file : str = "/data/elevation/ETOPO_2022_v1_60s_N90W180_surface.nc",
                                        coast_shapefile : str = "/data/shapefiles/ne_10m_coastline/ne_10m_coastline.shp",
                                        verbose : bool = False):
    if verbose : print(f'-> Adding external variables {variables}.')
    df = df_orig.copy()
    # check that latitude and longitudes are defined symetrically around 0° 
    if any(df[lat])>90:
        df[lat]=utils.convert_lat_0_180_to_neg90_90(np.array(df[lat].values))
    if any(df[lon])>180 :
        df[lon]=utils.convert_lon_0_360_to_neg180_180(np.array(df[lon].values))
    
    # add columns with the variables specified in argument
    if 'ele' in variables :
        if verbose : print(f'   > elevation data will be taken from file {dem_file}')
        df = interpolate_dem_at_latlon_points(df,dem_file = dem_file)
    if 'D' in variables:
        if verbose : print(f'   > coastlines will be taken from file {coast_shapefile}')
        df = compute_distance_to_coast(df,coast_shapefile,lat=lat,lon=lon)
    return df

def interpolate_dem_at_latlon_points(df_latlon: pd.DataFrame, dem_file :str ="/data/elevation/ETOPO_2022_v1_60s_N90W180_surface.nc",lat='lat',lon='lon'):
    """ TODO 
    lat : -90 to 90
    lon : -180 to 180
    """
    df = df_latlon.copy()
    if any(df[lon]>180):
        df[lon]=utils.convert_lon_0_360_to_neg180_180(np.array(df[lon]))
    if any(df[lat]>90):
        df[lat]=utils.convert_lat_0_180_to_neg90_90(np.array(df[lat]))

    # Load global elevation grid (ETOPO for instance)
    ds = xr.open_dataset(utils.get_project_root()+dem_file,engine='netcdf4') # ds must have lat/lon coords and z variable
    lats_dem = ds[lat].values
    lons_dem = ds[lon].values
    elevation_grid = ds['z'].values

    # interpolation at df points
    interp_elev = RegularGridInterpolator(
        (lats_dem, lons_dem),
        elevation_grid,
        bounds_error=False,
        fill_value=np.nan
    )

    lats_df = df[lat].values
    lons_df = df[lon].values

    elev_at_obs = interp_elev(np.column_stack([lats_df, lons_df])) # type: ignore
    df['ele'] = elev_at_obs

    # we seen there are points at -6000m (indonesia, in the ocean). This is a problem since I suppose that itrace simulation only models terrestrial points, but there was a buffer wround the coast that included some points in the ocean.
    # either i must manage to mask these points, or clip the values to sea level (or -423m, the lowest surface elevation?)
    df.loc[df['ele']<0,'ele']=0
    df = df.dropna(subset=['ele'])
    
    return df

def compute_distance_to_coast(df_latlon, coast_shp_path="/data/shapefiles/ne_10m_coastline/ne_10m_coastline.shp",lat='lat',lon='lon'):
    """
    Add geodesic distance to nearest coastline (in meters) to a dataframe 
    containing columns lat and lon.
    Inputs:
        df : pandas.DataFrame, it must contain lat and lon columns.
        lat : str
        lon : str
        coast_shp_path : str, the path to a coastline shapefile.
    Outputs:
        df : pandas.DataFrame, same dataframe as input but with an extra column 'D'.
    """
    df = df_latlon.copy()

    if any(df[lon]>180):
        df[lon] = utils.convert_lon_0_360_to_neg180_180(df[lon].values)
    if any(df[lat]>90):
        df[lat] = utils.convert_lat_0_180_to_neg90_90(df[lat].values)
    
    coast = gpd.read_file(utils.get_project_root()+coast_shp_path).to_crs("EPSG:4326")

    geod = Geod(ellps="WGS84")

    distances = []
    for _, row in df.iterrows():
        pt = Point(row[lon], row[lat])
        # find actual nearest geometry, not just BB neighbors
        nearest_idx = coast.sindex.nearest(pt)[1]
        nearest_geom = coast.geometry.iloc[nearest_idx]

        # compute geodesic distance
        p1, p2 = nearest_points(pt, nearest_geom)
        _, _, dist = geod.inv(p1.iloc[0].x, p1.iloc[0].y, p2.iloc[0].x, p2.iloc[0].y) # type: ignore
        distances.append(dist)

    df["D"] = distances # dist to coast in m
    return df