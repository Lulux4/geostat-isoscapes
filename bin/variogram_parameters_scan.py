from geostat_isoscapes_tools import geostat_utils as gutils, plot_utils as putils, sisal_utils as sutils, utils
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import os 
sns.set_style('dark')

###############################################
# Define parameters to apply successively
###############################################

trends = [
    'multiple_linear_latabs_latReLU',
    'multiple_linear_latabs_latReLU_P',
    'multiple_linear_latabs_latReLU_D',
    'multiple_linear_latabs_latReLU_ele',
    'multiple_linear_latabs_latReLU_P_D',
    'multiple_linear_latabs_latReLU_ele_P',
    'multiple_linear_latabs_latReLU_ele_D',
    'multiple_linear_latabs_latReLU_ele_P_D'
    ]

azimuths = [None] # 0 45 90 180

# Config
params = {
        # 'trend': 'xxxxxx' to be set in the loop 
        'trend_before_mask': True,
        'maxlag': 1.2e7,
        'nlags': 20,
        'centers': None,
        'tolerance' : 22.5,
        'model_name' : 'spherical',
        'mask radius [km]': 1000,
        'res [months]': 12*200,
        'Itrace simulation spec' : {'time':'20kyrBP',},
        'countries':None
        }
data_cols = {'lat':'lat',
            'lon':'lon',
            'quantity':'d18Op'}

verbose =  False

#### if loop only on trend i can load the data just once
# Load sisal data 
sisal_df = gutils.get_sisal_data_for_kriging(res=int(params['res [months]']/12),countries=params['countries'],verbose=verbose)
# load itrace data
itrace_data = gutils.get_preprocessed_itrace_data(res=params['res [months]'],P=True,format='xr',verbose=False,
                                            sim_prefix = 'b.e13.Bi1850C5.f19_g16',
                                            sim_suffix ='100001-199912', #12:'800001-899912', #16:'400001-499912', #20 :'100001-199912',
                                            sim_forcings='ice_ghg_orb_wtr',
                                            sim_kyr= int(params['Itrace simulation spec']['time'][:2]),
                                            sim_num='01',
                                            sim_model='clm2.h0',
                                            ) # xr
for trend in trends:
    print('..........................................')
    print('trend=',trend)
    print('..........................................')

    params['trend']=trend
    fp = f"{utils.get_project_root()}/output/variograms/itrace/sim{params['Itrace simulation spec']['time']}/res{params['res [months]']}/maxlag{str(int(params['maxlag']))}nlags{params['nlags']}/trend_{params['trend']}/"
    if not os.path.exists(fp) :
        os.makedirs(fp)

    # Variogram computation
    for azimuth in azimuths :
        print(f'computation dir={azimuth}')
        params['direction']=azimuth
        gutils.iterate_and_aggregate_variograms(data_ds = itrace_data, #type:ignore
                                            fp=fp,
                                            config_dict=params,
                                            data_cols=data_cols,
                                            mask_pts = sisal_df, # type:ignore
                                            verbose = verbose
                                        )
    # Variogram model fitting and plotting (saving results only)
    dict_dfs = {}
    for a in azimuths:
        dict_dfs[f'i{a}'] = {'df': pd.read_csv(fp+f'vario_{a}_df.csv')}

    for key in dict_dfs.keys():
        dict_dfs[key]['params'],dict_dfs[key]['fct_fitted'],dict_dfs[key]['pcov'] = gutils.fit_variogram_model(bins=dict_dfs[key]['df']['lag'], # type:ignore
                                                                                    gammas=dict_dfs[key]['df']['gamma'],
                                                                                    model_name=params['model_name'],
                                                                                    pair_counts=dict_dfs[key]['df']['count']
                                                                                    ) 
        
    fig,ax = putils.plot_variogram_from_bins_and_gamma(centers=dict_dfs['iNone']['df']['lag'],
                                            gamma=dict_dfs['iNone']['df']['gamma'],
                                            time='iTrace dataset - 11.7 to 11 ky BP - weighted average',
                                            counts=dict_dfs['iNone']['df']['count'],
                                            min_pairs=20,
                                            plot_model=True,
                                            model_name=params['model_name'],
                                            model_fct=dict_dfs['iNone']['fct_fitted'],
                                            model_params=dict_dfs['iNone']['params'],
                                            figsize=(10,5),
                                            save_name=fp+'fig.png'
                                            ) # type:ignore

    list_dir = [az for az in azimuths if az is not None]
    if len(list_dir)>0:
        fig,axes = plt.subplots(2,2,figsize=(20,10))
        for i in range(2):
            for j in range(2):
                key = f'i_{list_dir[j+2*i]}'
                axes[i,j] = putils.plot_variogram_from_bins_and_gamma(centers=dict_dfs[key]['df']['lag'],
                                                                    gamma=dict_dfs[key]['df']['gamma'],
                                                                    time='iTrace dataset - 11.7 to 11 ky BP - weighted average',
                                                                    counts=dict_dfs[key]['df']['count'],
                                                                    min_pairs=20,
                                                                    plot_model=True,
                                                                    model_name=params['model_name'],
                                                                    model_fct=dict_dfs[key]['fct_fitted'],
                                                                    model_params=dict_dfs[key]['params'],
                                                                    figsize=(10,5),
                                                                    ax_ = axes[i,j],
                                                                    save_name=f'{fp}fig_dir{list_dir[j+2*i]}.png'
                                                                    ) # type:ignore
                axes[i,j].set_title(f'dir={list_dir[j+2*i]}')
        fig.tight_layout()
        plt.savefig(f'{fp}fig_dirs_all.png')
    plt.close('all')