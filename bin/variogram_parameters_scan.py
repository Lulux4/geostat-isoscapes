from geostat_isoscapes_tools import geostat_utils as gutils, plot_utils as putils, sisal_utils as sutils, utils
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import os 
import itertools
sns.set_style('dark')

###############################################
# Define parameters to apply successively
###############################################
# WHICH DATASET DO YOU WANT TO PROCESS ?
sisal = True
itrace = True
# DEFINE THE PARAMETERS ON WHICH TO LOOP :
trends = [
    'multiple_linear_ele_D',
    'multiple_linear_D',    
    'multiple_linear_latabs_latReLU',
    # 'multiple_linear_latabs_latReLU_P', # /!/ P cannot be retrieved for sisal data
    'multiple_linear_latabs_latReLU_D',
    'multiple_linear_latabs_latReLU_ele',
    # 'multiple_linear_latabs_latReLU_P_D', # /!/ P cannot be retrieved for sisal data
    # 'multiple_linear_latabs_latReLU_ele_P', # /!/ P cannot be retrieved for sisal data
    'multiple_linear_latabs_latReLU_ele_D',
    # 'multiple_linear_latabs_latReLU_ele_P_D' # /!/ P cannot be retrieved for sisal data
    ]

azimuths = [None] # 0 45 90 180

sims = {'12':{'num':'05','suffix':'800001-899912'},
        '16':{'num':'01','suffix':'400001-499912'},
        '20':{'num':'01','suffix':'100001-199912'}
        }

res_months = 12*200 # min 12 if using sisal, min 1 if using itrace

verbose = False
regions_list = [('continents',['South America','North America']),
                ('continents',['Europe']),
                ('continents',['Asia']),
                ('continents',['Oceania'])] #[None]#[('continents',['South America'])]#('subregion',['Western Europe','Southern Europe','Northern Europe'])
buffer_km = 0 # /!/ continent europe -> set buffer to 0 otherwise it yields pb with antimeridional line
model_name = 'spherical'

# Define name of cols 
data_cols = {'lat':'lat',
            'lon':'lon',
            'quantity':'d18Op'}
cols_sisal = ['binned_age','lat','lon','site_id','d18Op_VSMOW_exactconv']

# Prepare the iterations ###############################
iters = itertools.product(sims.keys(),trends,azimuths,regions_list)

tmp_kyr = None
tmp_regions = 'init'

for kyr,trend,azimuth,regions in iters :
    print('=====================================================================')
    print(f'Time={kyr} kyr BP, trend={trend}, direction={azimuth}, regions={regions}')
    print('=====================================================================')
    # COMFIGURATION ####################################
    sisal_params = {
        'trend_before_mask': True,
        'maxlag': 1.2e7,
        'nlags': 20,
        'centers': None,
        'trend':trend,
        'tolerance' : 22.5,
        'model_name' : model_name,
        'mask radius [km]': 1000,
        'res [months]': res_months,
        'direction': azimuth,
        'regions':regions,
        'buffer_km': buffer_km,
        'const_coords':False,}
    
    itrace_params = sisal_params.copy()
    itrace_params['Itrace simulation spec'] = { 'kyr': kyr,                   
                                                'num': sims[kyr]['num'],                
                                                'model':'clm2.h0',
                                                'forcings':'ice_ghg_orb_wtr',   
                                                'suffix': sims[kyr]['suffix'], 
                                                'prefix':'b.e13.Bi1850C5.f19_g16'}
    
    fp_itrace = f"{utils.get_project_root()}/output/variograms/itrace/sim{itrace_params['Itrace simulation spec']['kyr']}kyrBP/res{itrace_params['res [months]']}/maxlag{str(int(itrace_params['maxlag']))}nlags{itrace_params['nlags']}/trend_{itrace_params['trend']}/"
    fp_sisal = f"{utils.get_project_root()}/output/variograms/sisal/slice{itrace_params['Itrace simulation spec']['kyr']}kyrBP/res{sisal_params['res [months]']}/maxlag{str(int(sisal_params['maxlag']))}nlags{sisal_params['nlags']}/trend_{sisal_params['trend']}/"
    if regions is not None :
        str_regions = ''
        for r in regions[1]:
            str_regions += r
        fp_itrace = f"{str(fp_itrace)}{regions[0]}_{str_regions.replace(' ','')}/"
        fp_sisal = f"{str(fp_sisal)}{regions[0]}_{str_regions.replace(' ','')}/"

    if not os.path.exists(fp_itrace) :
        os.makedirs(fp_itrace)
    if not os.path.exists(fp_sisal) :
        os.makedirs(fp_sisal)

    # DATA LOADING #######################################
    if (tmp_regions == 'init') or (regions!=tmp_regions): # avoid to reload the same data as previous iteration
        tmp_regions = regions
        sisal_df = gutils.get_sisal_data_for_kriging(res=int(sisal_params['res [months]' ]/12),
                                                     regions=regions,
                                                     buffer_km=sisal_params['buffer_km'],
                                                     verbose=verbose)

    if (tmp_kyr is None) or (kyr!=tmp_kyr) : #->if kyr is the same as previous iteration, do not reload the data
        tmp_kyr = kyr
        # sisal data must be truncated to the right time period
        sisal_df_valid = sisal_df.loc[(sisal_df['binned_age']>(int(kyr)-1)*1000)&(sisal_df['binned_age']<=int(kyr)*1000),cols_sisal].rename(columns={'binned_age':'time','d18Op_VSMOW_exactconv':'d18Op'}).copy() #type:ignore
        # load itrace data for the given kyr
        itrace_data = gutils.get_preprocessed_itrace_data(
            res=itrace_params['res [months]'],
            P=True,
            format='xr',
            verbose=verbose,
            regions=regions,
            buffer_km = itrace_params['buffer_km'],
            sim_prefix = itrace_params['Itrace simulation spec']['prefix'],
            sim_suffix =itrace_params['Itrace simulation spec']['suffix'], 
            sim_forcings=itrace_params['Itrace simulation spec']['forcings'],
            sim_kyr= int(itrace_params['Itrace simulation spec']['kyr']),
            sim_num=itrace_params['Itrace simulation spec']['num'],
            sim_model=itrace_params['Itrace simulation spec']['model'],
        ) # xr
    # VARIOGRAMS ##########################################
    if itrace :
        print('variogram itrace...')
        gutils.iterate_and_aggregate_variograms(data = itrace_data, #type:ignore
                                                fp=fp_itrace,
                                                config_dict=itrace_params,
                                                data_cols=data_cols,
                                                mask_pts = sisal_df, # type:ignore
                                                verbose = verbose
                                            )
    if sisal :
        print('variogram sisal...')
        gutils.iterate_and_aggregate_variograms(data = sisal_df_valid, #type:ignore
                                                fp=fp_sisal,
                                                config_dict=sisal_params,
                                                data_cols=data_cols,
                                                mask_pts = None,
                                                verbose = False
                                            )
    # Variogram model fitting and plotting (saving results only)
    dict_dfs = {}
    for a in azimuths:
        if itrace:
            dict_dfs[f'i{a}'] = {'df': pd.read_csv(fp_itrace+f'vario_{a}_df.csv')}
        if sisal :
            dict_dfs[f's{a}'] = {'df': pd.read_csv(fp_sisal+f'vario_{a}_df.csv')}
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
        print('plotting and saving figure')
        fig,ax = putils.plot_variogram_from_bins_and_gamma(centers=dict_dfs[key]['df']['lag'],
                                                gamma=dict_dfs[key]['df']['gamma'],
                                                time=f'{title} - {int(kyr)-1} to {kyr} kyr BP',
                                                counts=dict_dfs[key]['df']['count'],
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