from geostat_isoscapes_tools import geostat_utils as gutils, plot_utils as putils, sisal_utils as sutils, utils
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import os 
import itertools
import json

sns.set_style('dark')

###############################################
# Define parameters to apply successively
###############################################

# DATASETS TO STUDY :
sisal = True
itrace = False

# PARAMETERS LISTS :

mask_itrace_around_sisal_pts = False
mask_radius = 10

trends = [
    # 'multiple_linear_ele_D',
    # 'multiple_linear_D',    
    # 'multiple_linear_ele',
    # 'multiple_linear_lat_',
    # 'multiple_linear_lat_ele',
    # 'multiple_linear_lat_D',
    # 'multiple_linear_lat_ele_D',
    
    # 'multiple_linear_latabs_latReLU',
    # 'multiple_linear_latabs_latReLU_P', # /!/ P cannot be retrieved for sisal data
    # 'multiple_linear_latabs_latReLU_D',
    # 'multiple_linear_latabs_latReLU_ele',
    # 'multiple_linear_latabs_latReLU_P_D', # /!/ P cannot be retrieved for sisal data
    # 'multiple_linear_latabs_latReLU_ele_P', # /!/ P cannot be retrieved for sisal data
    'multiple_linear_latabs_latReLU_ele_D',
    # 'multiple_linear_latabs_latReLU_ele_P_D' # /!/ P cannot be retrieved for sisal data
    ]

azimuths = [None] # 0 45 90 180 # for directional variograms

sims = { # simulation spec for itrace dataset
    '1':{} # dummy entry for sisal only, to get first 1000 years BP
    # '12':{'num':'05','suffix':'800001-899912'},
    # '16':{'num':'01','suffix':'400001-499912'},
    # '20':{'num':'01','suffix':'100001-199912'}
    }

res_months_list = [12*1000] # min resolution is 12 months if using sisal, 1 month if using itrace

verbose = False

# geographical regions on which to mask itrace/sisal data for the variogram computations
# default is [None], otherwise list[ tuple( str,list[str] ) ] such as [('subregion',['Western Europe','Southern Europe']),.....]
regions_list = [
    None
    # ('continents',['South America','North America','Europe','Asia','Oceania','Africa'])
    # ('countries',['United States of America','Mexico'])
    # ('continents',['South America']),
    # ('continents',['North America']),
    # ('subregions',['Nothern Europe','Southern Europe','Western Europe']),
    # ('continents',['Asia']),
    # ('continents',['Oceania']),
    # ('continents', ['Africa']),
    # ('countries',['China']),
    # ('countries',['United States of America']),
    # ('countries',['Canada']),
    # ('countries',['Australia']),
    # ('countries',['Brazil']),
    # ('countries',['France']),
    # ('countries',['China']),
    # ('countries',['Bolivia']),
    # ('countries',['South Africa']),
    # ('countries',['Afghanistan'])
    ] 
# buffer to keep some data around the regions boundaries (typically for country-sized regions) 
buffer_km = 0 # /!/ continent europe -> set buffer to 0 otherwise it yields pb with antimeridional line

# Variogram model to try to fit
model_name = 'spherical'
maxlag = 12e6
nlags = 10

# Naming convention of columns
data_cols = {'lat':'lat',
            'lon':'lon',
            'quantity':'d18Op'}
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
        'kyr' : kyr,
        'trend_before_mask': True,
        'maxlag': maxlag,
        'nlags': nlags,
        'centers': None,
        'trend':trend,
        'tolerance' : 22.5,
        'model_name' : model_name,
        'mask radius [km]': mask_radius,
        'res [months]': res_months,
        'direction': azimuth,
        'regions':regions,
        'buffer_km': buffer_km,
        'const_coords':False,}
    
    if itrace : 
        itrace_params = params.copy()
        itrace_params['Itrace simulation spec'] = { 'kyr': kyr,                   
                                                'num': sims[kyr]['num'],                
                                                'model':'clm2.h0',
                                                'forcings':'ice_ghg_orb_wtr',   
                                                'suffix': sims[kyr]['suffix'], 
                                                'prefix':'b.e13.Bi1850C5.f19_g16'}
    
        fp_itrace = f"{utils.get_project_root()}/output/variograms/itrace/sim{itrace_params['Itrace simulation spec']['kyr']}kyrBP/res{itrace_params['res [months]']}/maxlag{str(int(itrace_params['maxlag']))}nlags{itrace_params['nlags']}/trend_{itrace_params['trend']}/"
        if mask_itrace_around_sisal_pts :
            fp_itrace = fp_itrace + f"sisal_mask_{mask_radius}km/"
        else : 
            fp_itrace = fp_itrace + "no_mask/"
    if sisal : 
        fp_sisal = f"{utils.get_project_root()}/output/variograms/sisal/slice{params['kyr']}kyrBP/res{params['res [months]']}/maxlag{str(int(params['maxlag']))}nlags{params['nlags']}/trend_{params['trend']}/"
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
        print('----- New sisal resolution : loading sisal global df')
        sisal_df_global = gutils.get_sisal_data_for_kriging(res=int(res_months/12),
                                                                    regions=None,
                                                                    buffer_km=buffer_km,
                                                                    verbose=verbose)
        sisal_df = sisal_df_global
        sisal_change = True
        init=False

    # 2. If region changed, truncate the global df 
    if sisal & ((regions!=tmp_regions) | sisal_change) & (regions is not None) : 
        print('----- New sisal upper-level param res was set, or new regions params is detected : applying a new mask to sisal global df')
        sisal_df_regions = utils.mask_regions_shape(sisal_df_global,buffer_km=buffer_km,regions=regions) # type:ignore
        sisal_df = sisal_df_regions
        sisal_change = True

    # 3. If the temporal slice (kyr) is not the same as previous iteration, 
    #    we just need to take the right slice of sisal data (no reload)
    if sisal & (((tmp_kyr is None)|((kyr!=tmp_kyr))|(tmp_res!=res_months)) | sisal_change) :
        print('----- New sisal upper-level params region or res was set, or new temporal slice param detected : taking the new slice from sisal df')
        sisal_df_valid = sisal_df.loc[(sisal_df['binned_age']>(int(kyr)-1)*1000)&(sisal_df['binned_age']<=int(kyr)*1000),cols_sisal].rename(columns={'binned_age':'time','d18Op_VSMOW_exactconv':'d18Op'}).copy() #type:ignore
        
    # 4. For itrace, we need to reload in any case since we cannot load several slices at the smae time
    itrace_change = False
    if itrace & ((tmp_kyr is None) or (kyr!=tmp_kyr) or (tmp_res!=res_months)): 
        print('----- New itrace params res or kyr : loading appropriate data')
        itrace_data_global = gutils.get_preprocessed_itrace_data(
            res=itrace_params['res [months]'],
            P=True,
            format='xr',
            verbose=verbose,
            regions=None,
            buffer_km = itrace_params['buffer_km'],
            sim_prefix = itrace_params['Itrace simulation spec']['prefix'],
            sim_suffix =itrace_params['Itrace simulation spec']['suffix'], 
            sim_forcings=itrace_params['Itrace simulation spec']['forcings'],
            sim_kyr= int(itrace_params['Itrace simulation spec']['kyr']),
            sim_num=itrace_params['Itrace simulation spec']['num'],
            sim_model=itrace_params['Itrace simulation spec']['model'],
        ) # xr
        itrace_data = itrace_data_global
        itrace_change = True

    # 4. Apply new geographical mask if the regions have changed from previous iteration, or if it is the first iter.
    if (itrace_change | ((regions!=tmp_regions)& itrace)) & (regions is not None): 
        print('----- New itrace params res or kyr was set, or new regions detected, so we re-mask the global data for the specified regions')
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
                                                mask_pts = sisal_df if mask_itrace_around_sisal_pts else None, # type:ignore # TODO : sisal_df_valid or region? depends on the needs, be careful
                                                verbose = verbose
                                            )
    if sisal :
        print('variogram sisal...')
        gutils.iterate_and_aggregate_variograms(data = sisal_df_valid, #type:ignore
                                                fp=fp_sisal,
                                                config_dict=params,
                                                data_cols=data_cols,
                                                mask_pts = None,
                                                verbose = False
                                            )
    # Variogram model fitting and plotting (saving results only)
    dict_dfs = {}
    for a in azimuths:
        if itrace :
            if os.path.exists(fp_itrace+f'vario_{a}_df.csv'): # type: ignore
                dict_dfs[f'i{a}'] = {'df': pd.read_csv(fp_itrace+f'vario_{a}_df.csv')} # type: ignore
        if sisal :
            if os.path.exists(fp_sisal+f'vario_{a}_df.csv'): # type: ignore
                dict_dfs[f's{a}'] = {'df': pd.read_csv(fp_sisal+f'vario_{a}_df.csv')} # type: ignore
    for key in dict_dfs.keys():
        dict_dfs[key]['params'],dict_dfs[key]['fct_fitted'],dict_dfs[key]['pcov'] = gutils.fit_variogram_model(bins=dict_dfs[key]['df']['lag'], # type:ignore
                                                                                    gammas=dict_dfs[key]['df']['gamma'],
                                                                                    model_name=model_name,
                                                                                    pair_counts=dict_dfs[key]['df']['count']
                                                                                    ) 
        if key.startswith('i'):
            fp = fp_itrace
            title = 'iTrace dataset'
            textbox_loc=(0.835, 0.25)
            legend_loc='lower right'
        else : 
            fp = fp_sisal
            title = 'SISAL dataset'
            textbox_loc=(0.835, 0.65)
            legend_loc='upper right'

        print('saving vario parameters')
        with open(f'{fp}variogram_params_{key[1:]}.json', 'w') as filename:
            json.dump(dict_dfs[key]['params'], filename)

        print('plotting and saving figure')
        fig,ax = putils.plot_variogram_from_bins_and_gamma(centers=dict_dfs[key]['df']['lag'].values,
                                                gamma=dict_dfs[key]['df']['gamma'].values,
                                                time=f'{title} - {int(kyr)-1} to {kyr} kyr BP',
                                                counts=dict_dfs[key]['df']['count'].values,
                                                min_pairs=20,
                                                plot_model=True,
                                                model_name=model_name,
                                                model_fct=dict_dfs[key]['fct_fitted'],
                                                model_params=dict_dfs[key]['params'],
                                                figsize=(10,5),
                                                textbox_loc=textbox_loc,
                                                legend_loc=legend_loc,
                                                save_name=f'{fp}fig_{key[1:]}.png'
                                                ) # type:ignore

    plt.close('all')