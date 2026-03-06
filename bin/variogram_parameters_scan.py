from geostat_isoscapes_tools import geostat_utils as gutils, plot_utils as putils, sisal_utils as sutils, utils
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import os 
import itertools
import json
import numpy as np

sns.set_style('dark')

# ==========================================================================================
# CONFIGURATION
# ==========================================================================================
# NAME OF EXPERIMENT
exp_name = '2026-02-24_sisal_vs_itrace'

# DATASET(S) OF INTEREST FOR THE VARIOGRAPHY :
sisal = True
itrace = True

# PARAMETERS LISTS :
mask_itrace_around_sisal_pts = True # variography on itrace masked at sisal points?
mask_radius = 2000

trends = [
    # 'multiple_linear_ele_D',
    # 'multiple_linear_D',    
    # 'multiple_linear_ele',
    # 'multiple_linear_lat_',
    # 'multiple_linear_lat_ele',
    # 'multiple_linear_lat_D',
    # 'multiple_linear_lat_ele_D',
    # None
    # 'multiple_linear_lat_latquad',
    # 'multiple_linear_lat_latquad_P', # /!/ P cannot be retrieved for sisal data
    # 'multiple_linear_lat_latquad_D',
    # 'multiple_linear_lat_latquad_ele',
    # 'multiple_linear_lat_latquad_P_D', # /!/ P cannot be retrieved for sisal data
    # 'multiple_linear_lat_latquad_ele_P', # /!/ P cannot be retrieved for sisal data
    'multiple_linear_lat_latquad_ele_D',
    # 'multiple_linear_lat_latquad_ele_P_D' # /!/ P cannot be retrieved for sisal data
    # 'multiple_linear_latabs_latReLU',
    # 'multiple_linear_latabs_latReLU_P', # /!/ P cannot be retrieved for sisal data
    # 'multiple_linear_latabs_latReLU_D',
    # 'multiple_linear_latabs_latReLU_ele',
    # 'multiple_linear_latabs_latReLU_P_D', # /!/ P cannot be retrieved for sisal data
    # 'multiple_linear_latabs_latReLU_ele_P', # /!/ P cannot be retrieved for sisal data
    # 'multiple_linear_latabs_latReLU_ele_D',
    # 'multiple_linear_latabs_latReLU_ele_P_D'
    
    ]

azimuths = [None] # 0,45,90,180] # 0 45 90 180 # for directional variograms

# how to access itrace data ? Need the datafolder (not provided in my repo, seee intructions to download it) and the json (provided in repo data/ directory) that contains indications on the filenames.
itrace_folder = "/media/luluxette/T7_Shield/pdm/iTrace/" # itrace data folder, (absolute path)
with open(f'{utils.get_project_root()}/data/iTrace_simulations_dict.json', 'r') as f:
    itrace_sims = json.load(f)

# keep only the simulations of interest (each key represent a 1000 yr slice)
sims = dict((k,itrace_sims[k]) for k in (
    '12',
    '13',
    '14',
    '15',
    '16',
    '17',
    '18',
    '19',
    '20'
    ) if k in itrace_sims)

# If not computing on itrace, define instead a dummy key for accessing a sisal slice or all times
# e.g. : sims = {'1':{}}  for the slice 0-1000 yr BP or sims={'all':{} } for all time points
# sims = {'all':{}}

res_months_list = [12*200] # min resolution is 12 months if using sisal, 1 month if using itrace
save_all=  True # whether to save all variograms data (not only the aggregated one over 1000yr).
verbose = False

# Geographical regions on which to mask itrace/sisal data for the variogram computations
# default is [None], otherwise list[ tuple( str,list[str] ) ] such as [('subregion',['Western Europe','Southern Europe']),.....]
regions_list = [
    None
    # ('continents',['South America','North America','Europe','Asia','Oceania','Africa'])
    ] 
# Buffer to keep some data around the regions boundaries (typically for country-sized regions) 
buffer_km = 0 # /!/ continent europe -> set buffer to 0 otherwise it yields pb with antimeridional line

# Variography parameters
model_name = 'spherical'
maxlag = 10e6
nlags = 10
bin_func = 'even'
weighting = 'lags'
weights_power = 0.5

if not ('+' in model_name) : # bounds for a simple model (spherical, gaussian, exp)
    model_bounds = (
        np.array([
            0.0,   # min range
            0.0,   # min sill
            0.2]), # min nugget  : here you can force the model to have non-zero nugget
        np.array([
            maxlag,# max range 
            50,    # max sill
            10     # max nugget
        ])
    )  # max nugget 
else :
    model_bounds = ( # bounds for composite model (two models, 5 parameters)
        np.array([
            1.0,   # min nugget
            0.0,   # min range1
            0.0,   # min sill1
            0.0,   # min range2
            0.0,   # min sill2
        ]),
        np.array([
            10,    # max nugget
            maxlag,# max range1
            50,    # max sill1
            maxlag,# max range2
            50     # max sill2
        ])
    )

# Naming convention of columns
data_cols = {'lat':'lat','lon':'lon','quantity':'d18Op'}
cols_sisal = ['binned_age','lat','lon','site_id','d18Op_VSMOW_exactconv']

# Prepare the iterations 
iters = itertools.product(res_months_list,sims.keys(),regions_list,trends,azimuths)

tmp_kyr = None
tmp_regions = None
tmp_res = None
init = True # turns to false after first iteration

for res_months,kyr,regions,trend,azimuth in iters :
    print('====================================================================================================================')
    print(f'Time={kyr} kyr BP, resolution={res_months} months, trend={trend}, direction={azimuth}, regions={regions}')
    print('====================================================================================================================')
    # ======================= CONFIGURATION =======================
    params = {
        'exp_name' :exp_name,
        'kyr' : kyr,
        'trend_before_mask': True,
        'maxlag': maxlag,
        'nlags': nlags,
        'bin_func':bin_func,
        'weighting':weighting,
        'weights_power':weights_power,
        'centers': None,
        'trend':trend,
        'tolerance' : 22.5,
        'model_name' : model_name,
        'model_bounds': model_bounds,
        'mask radius [km]': mask_radius,
        'plot_mask':True,
        'res [months]': res_months,
        'direction': azimuth,
        'regions':regions,
        'buffer_km': buffer_km,
        'const_coords':False,}
    
    if itrace : 
        itrace_params = params.copy()
        itrace_params['Itrace simulation spec'] = { 'kyr_filename': sims[kyr]['yrBPname'],
                                                'kyr': kyr,                   
                                                'num': sims[kyr]['num'],                
                                                'model':'clm2.h0',
                                                'forcings':sims[kyr]['forcings'],   
                                                'suffix': sims[kyr]['suffix'], 
                                                'prefix':'b.e13.Bi1850C5.f19_g16'}
    
        fp_itrace = f"{utils.get_project_root()}/output/variograms/{params['exp_name']}/itrace/sim{itrace_params['Itrace simulation spec']['kyr']}kyrBP/res{itrace_params['res [months]']}/maxlag{str(int(itrace_params['maxlag']))}nlags{itrace_params['nlags']}/trend_{itrace_params['trend']}/"
        if mask_itrace_around_sisal_pts :
            fp_itrace = fp_itrace + f"sisal_mask_{mask_radius}km/"
        else : 
            fp_itrace = fp_itrace + "no_mask/"
    if sisal : 
        fp_sisal = f"{utils.get_project_root()}/output/variograms/{params['exp_name']}/sisal/slice{params['kyr']}kyrBP/res{params['res [months]']}/maxlag{str(int(params['maxlag']))}nlags{params['nlags']}/trend_{params['trend']}/"
    if regions is not None :
        str_regions = ''
        for r in regions[1]: #type:ignore
            str_regions += r
        if itrace : 
            fp_itrace = f"{str(fp_itrace)}{regions[0]}_{str_regions.replace(' ','')}/"
        if sisal :
            fp_sisal = f"{str(fp_sisal)}{regions[0]}_{str_regions.replace(' ','')}/"
    if itrace :
        if not os.path.exists(fp_itrace):
            os.makedirs(fp_itrace)
    if sisal :
        if not os.path.exists(fp_sisal) :
            os.makedirs(fp_sisal)

    # ================= DATA LOADING =====================
    # set a change "tracker" to avoid repeating if statements
    sisal_change = False
    # 1. If the resolution changed, we must reload sisal global data
    if (sisal & (tmp_res!=res_months)) | (mask_itrace_around_sisal_pts & init):
        # print('----- New sisal resolution : loading sisal global df')
        sisal_df_global = gutils.get_sisal_data_for_kriging(res=int(res_months/12),
                                                            temp_ds_name='itrace',
                                                            temp_ds_path=itrace_folder,
                                                            regions=None,
                                                            buffer_km=buffer_km,
                                                            verbose=verbose)
        sisal_df = sisal_df_global
        sisal_change = True
        init=False # even if res changes again, we will not need to reload sisal for adapting the mask
    # 2. If region changed, truncate the global df 
    if (sisal | mask_itrace_around_sisal_pts) & ((regions!=tmp_regions) | sisal_change) & (regions is not None) : 
        # print('----- New sisal upper-level param res was set, or new regions params is detected : applying a new mask to sisal global df')
        sisal_df_regions = utils.mask_regions_shape(sisal_df_global,buffer_km=buffer_km,regions=regions) # type:ignore
        sisal_df = sisal_df_regions
        sisal_change = True

    # 3. If the temporal slice (kyr) is not the same as previous iteration, 
    #    we just need to take the right slice of sisal data (no reload)
    if (sisal | mask_itrace_around_sisal_pts) & (((tmp_kyr is None)|((kyr!=tmp_kyr))|(tmp_res!=res_months)) | sisal_change) :
        # print('----- New sisal upper-level params region or res was set, or new temporal slice param detected : taking the new slice from sisal df')
        if kyr!='all':
            sisal_df_valid = sisal_df.loc[(sisal_df['binned_age']>=(int(kyr)-1)*1000)&(sisal_df['binned_age']<int(kyr)*1000),cols_sisal].rename(columns={'binned_age':'time','d18Op_VSMOW_exactconv':'d18Op'}).copy() #type:ignore
        else :
            sisal_df_valid = sisal_df.rename(columns={'binned_age':'time','d18Op_VSMOW_exactconv':'d18Op'}).copy() # type:ignore

    # 4. For itrace, we need to reload in any case since we cannot load several slices at the smae time
    itrace_change = False
    if itrace & ((tmp_kyr is None) or (kyr!=tmp_kyr) or (tmp_res!=res_months)): 
        # print('----- New itrace params res or kyr : loading appropriate data')
        itrace_data_global = gutils.get_preprocessed_itrace_data(
            data_folder=itrace_folder,
            res=itrace_params['res [months]'],
            P=True,
            format='xr',
            verbose=verbose,
            regions=None,
            buffer_km = itrace_params['buffer_km'],
            sim_prefix = itrace_params['Itrace simulation spec']['prefix'],
            sim_suffix =itrace_params['Itrace simulation spec']['suffix'], 
            sim_forcings=itrace_params['Itrace simulation spec']['forcings'],
            sim_kyr= int(itrace_params['Itrace simulation spec']['kyr_filename']),
            sim_num=itrace_params['Itrace simulation spec']['num'],
            sim_model=itrace_params['Itrace simulation spec']['model'],
        ) # xr
        itrace_data = itrace_data_global
        itrace_change = True

    # 4. Apply new geographical mask if the regions have changed from previous iteration, or if it is the first iter.
    if (itrace_change | ((regions!=tmp_regions)& itrace)) & (regions is not None): 
        # print('----- New itrace params res or kyr was set, or new regions detected, so we re-mask the global data for the specified regions')
        itrace_data = utils.mask_regions_shape(itrace_data_global,buffer_km=buffer_km,regions=regions) # type:ignore

    # 5. update our temporary values of kyr and res
    tmp_kyr = kyr
    tmp_res = res_months
    tmp_regions = regions

    # VARIOGRAMS ##########################################
    if itrace :
        print('variogram itrace...')
        gutils.iterate_and_aggregate_variograms(data = itrace_data, #type:ignore
                                                fp=fp_itrace,
                                                config_dict=itrace_params,
                                                data_cols=data_cols,
                                                mask_pts = sisal_df_valid if mask_itrace_around_sisal_pts else None, # type:ignore
                                                verbose = verbose,
                                                save_all=save_all
                                            )
    if sisal :
        print('variogram sisal...')
        gutils.iterate_and_aggregate_variograms(data = sisal_df_valid, #type:ignore
                                                fp=fp_sisal,
                                                config_dict=params,
                                                data_cols=data_cols,
                                                mask_pts = None,
                                                verbose = False,
                                                save_all=save_all
                                            )
    # Variogram model fitting and plotting (saving results only)
    dict_dfs = {}
    for a in azimuths:
        if itrace :
            if os.path.exists(fp_itrace+f'vario_{a}_df.csv'): # type: ignore
                dict_dfs[f'i{a}'] = {'df': pd.read_csv(fp_itrace+f'vario_{a}_df.csv')} # type: ignore
            # if save_all is True, then we should have more variograms df to load, all stored in df_all.csv 
            if os.path.exists(fp_itrace+f'vario_{a}_df_all_iterations.csv'): # type: ignore
                df_all = pd.read_csv(fp_itrace+f'vario_{a}_df_all_iterations.csv') # type: ignore
                for t in df_all['time_slice'].unique():
                    dict_dfs[f'i{a}_t{int(t)}'] = {'df': df_all.loc[df_all['time_slice']==t]}
        if sisal :
            if os.path.exists(fp_sisal+f'vario_{a}_df.csv'): # type: ignore
                dict_dfs[f's{a}'] = {'df': pd.read_csv(fp_sisal+f'vario_{a}_df.csv')} # type: ignore
            if os.path.exists(fp_sisal+f'vario_{a}_df_all_iterations.csv'): # type: ignore
                df_all = pd.read_csv(fp_sisal+f'vario_{a}_df_all_iterations.csv') # type: ignore
                for t in df_all['time_slice'].unique():
                    dict_dfs[f's{a}_t{t}'] = {'df': df_all.loc[df_all['time_slice']==t]}

    for key in dict_dfs.keys():
        dict_dfs[key]['params'],dict_dfs[key]['fct_fitted'],dict_dfs[key]['pcov'] = gutils.fit_variogram_model(bin_edges=dict_dfs[key]['df']['lag'].values, # type:ignore
                                                                                    gammas=dict_dfs[key]['df']['gamma'].values,
                                                                                    model_name=model_name,
                                                                                    bounds=params['model_bounds'],
                                                                                    weighting=params['weighting'],
                                                                                    weights_power=params['weights_power'],
                                                                                    pair_counts=dict_dfs[key]['df']['count'].values
                                                                                    ) 
        if key.startswith('i'):
            fp = fp_itrace
            title = 'iTrace dataset'
        else : 
            fp = fp_sisal
            title = 'SISAL dataset'

        print('saving vario parameters')
        with open(f'{fp}variogram_params_{key[1:]}.json', 'w') as filename:
            json.dump(dict_dfs[key]['params'], filename)
        
        if not('t' in key): # we do not want to save thousands of plots 
            print('plotting and saving figure')
            fig,ax = putils.plot_variogram_from_bins_and_gamma(bin_edges=dict_dfs[key]['df']['lag'].values,
                                                    gamma=dict_dfs[key]['df']['gamma'].values,
                                                    title=f'- {title} - {int(kyr)-1} to {kyr} kyr BP' if kyr !='all' else 'all times',
                                                    counts=dict_dfs[key]['df']['count'].values,
                                                    plot_model=True,
                                                    model_name=model_name,
                                                    model_fct=dict_dfs[key]['fct_fitted'],
                                                    model_params=dict_dfs[key]['params'],
                                                    save_name=f'{fp}fig_{key[1:]}.png'
                                                    ) # type:ignore

    plt.close('all')