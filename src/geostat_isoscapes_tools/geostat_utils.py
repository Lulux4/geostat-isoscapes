
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
# Useful functions for variogram computation using libraries
# =========================================================

def detrend_data(coords, vals):
    ''' detrend: fit plane z ~ x + y via a linear regression and subtract it to vals.
    Returns the residuals
    '''
    lr = LinearRegression().fit(coords, vals)
    trend = lr.predict(coords)
    resid = vals - trend
    return resid

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
                         trend_removal=False, 
                         nlags : int = 30, 
                         maxlag : float | str | None = 'median', 
                         centers : np.ndarray | None = None,
                         model : str ='spherical',
                         return_Variogram_object=False,
                         seed = 42
                         )-> skg.Variogram | tuple[np.ndarray,np.ndarray, np.ndarray, object] :
    """Compute experimental variogram with sampling.
    TODO : write this fct doc 
    """
    # check if the month contains too much samples. If so, sample it down to the sample_size.
    if len(df) > sample_size:
        print('downsampling')
        df = df.sample(sample_size, random_state=seed)
    
    vals = df[quantity].values
    
    if trend_removal :
        # Remove trend by fitting a plane 
        XY = np.column_stack((df['x'], df['y']))
        vals = detrend_data(XY,df[quantity].values)

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

# =========================================================================
# Iterative variogram computations
# =========================================================================

def iterative_variogram_computations(df,ref_bins = None,direction=None):
    """ TODO 
    """
    df['x'], df['y'] = project_coords(df['lon'].values, df['lat'].values, epsg="EPSG:3857")
    print('initializing loop...')
    bin_counts, gammas = [], []
    print('starting loop...')
    for t, g in tqdm(df.groupby('time')):
        if g['d18Op'].notna().sum() < 30:
            continue
        try:
            if ref_bins is None :
                b, g_exp, bin_count,_ = variogram_with_gstat(g,
                                                             quantity='d18Op',
                                                             trend_removal=True,
                                                             maxlag='median',
                                                             direction=direction,
                                                             return_Variogram_object=False) # type: ignore
                ref_bins = b
            else : 
                b, g_exp, bin_count,_ = variogram_with_gstat(g,
                                                             quantity='d18Op',
                                                             direction=direction,
                                                             trend_removal=True,
                                                             centers=ref_bins,
                                                             return_Variogram_object=False) # type: ignore    
            bin_counts.append(bin_count)
            gammas.append(g_exp)
                
        except Exception as e:
            print(f"Skipped {t}: {e}")
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