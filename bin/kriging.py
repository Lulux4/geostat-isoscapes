from geostat_isoscapes_tools import geostat_utils as gutils, plot_utils as putils, sisal_utils as sutils, utils
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import os 
import itertools
from tqdm import tqdm 
import json
import pandas as pd
sns.set_style('dark')

# ==================================
# Define the experimental setup
# ==================================

exp_name  = 'run_2026_02_25/allvario/alpha05/' # name of expermiment 

## vario_path = f'{utils.get_project_root()}/output/variograms/itrace/sim12kyrBP/res2400/maxlag8500000nlags15/trend_multiple_linear_latabs_latReLU_ele_D/no_mask/variogram_params_None.json'
## vario_path = f'{utils.get_project_root()}/output/variograms/itrace/sim12kyrBP/res12/maxlag12000000nlags20/trend_multiple_linear_latabs_latReLU_ele_D/sisal_mask/variogram_params_None.json'

# final results with the aggregated vario :
# vario_path = f'{utils.get_project_root()}/output/variograms/2026-02-24_sisal_vs_itrace/itrace/vario_res200y_maxlag10e6_nlags15_latlatquadeleD_mask2000.json'
# vario_subfolders = None
# or with the vario params at each time step:
vario_path = f'{utils.get_project_root()}/output/variograms/2026-02-24_sisal_vs_itrace/itrace/' #vario_res200y_maxlag10e6_nlags15_latlatquadeleD_mask2000.json
vario_subfolders = '/res2400/maxlag10000000nlags15/trend_multiple_linear_lat_latquad_ele_D/sisal_mask_2000km/'

alpha = 0.05 # max acceptable type I error rate for the Kleijnen test

temperature_ds_name = 'itrace' # 1) 'itrace' for 20-11 ka BP, or 2) 'krapp' for 0-800ka BP with res 1000 years
temperature_ds_path = '/media/luluxette/T7_Shield/pdm/iTrace/'

itrace_folder = '/media/luluxette/T7_Shield/pdm/iTrace/' # abs path to itrace data 

with open(f'{utils.get_project_root()}/data/iTrace_simulations_dict.json', 'r') as f:
    itrace_sims = json.load(f)
# Keep only the simulations of interest (the keys represent 1000 yr BP slices)
sims = dict((k,itrace_sims[k]) for k in ('12',
                                         '13',
                                         '14',
                                         '15',
                                         '16',
                                         '17',
                                         '18',
                                         '19',
                                         '20'
                                         ) if k in itrace_sims)

res_months_list = [12*200] # min resolution is 12 months

verbose = False

# Prepare the iterations 
iters = itertools.product(res_months_list,sims.keys())
tmp_kyr = None
tmp_res = None

for res_months,kyr in iters :
    print('====================================================================================================')
    print(f'Time={kyr} ka BP, resolution={res_months} months')
    print('====================================================================================================')
    # ======================= CONFIGURATION =======================
    params = {
        'kyr' : kyr,
        'trend_before_mask': True,
        'tolerance' : 22.5,
        'vario_path' : vario_path,
        'res [months]': res_months,
        }
    
    itrace_params = params.copy()
    itrace_params['Itrace simulation spec'] = { 
        'kyr': kyr,
        'kyr_filename':sims[kyr]['yrBPname'],                 
        'num': sims[kyr]['num'],                
        'model':'clm2.h0',
        'forcings':sims[kyr]['forcings'],   
        'suffix': sims[kyr]['suffix'], 
        'prefix':'b.e13.Bi1850C5.f19_g16'
    }
    fp_output = f"{utils.get_project_root()}/output/kriging/{exp_name}/res{itrace_params['res [months]']}/"
    if not os.path.exists(fp_output):
        os.makedirs(fp_output)
    
    # ================= DATASETS LOADING ====================================================
    # 1. If res changed, we reload sisal global data
    if (tmp_res!=res_months):
        print(f'..... Loading sisal dataset, resolution {res_months//12}')
        sisal_df = gutils.get_sisal_data_for_kriging(
            res=int(res_months//12),
            temp_ds_name=temperature_ds_name,
            temp_ds_path=temperature_ds_path,
            verbose=verbose
        )
        if not isinstance(sisal_df,pd.DataFrame) :
            raise TypeError('data to krige should be a pd df')
        sisal_df = sisal_df.rename(columns={'binned_age':'time','d18Op_VSMOW_exactconv':'d18Op'})
        # Remove Babylon cave as it is located at the exact same coordinates as Hollywood cave : pb for the kriging system
        sisal_df  = sisal_df[sisal_df.site_name != 'Babylon cave']
        # Remove DevilsHole cave as it seems to be wrongly estimated by itrace, leading to very high residuals at cross validation
        # sisal_df  = sisal_df[sisal_df.site_name != 'Devils Hole']

    # 2. Reload itrace whenever the time slice changes -- bc separate files --
    if (tmp_kyr is None) or (kyr!=tmp_kyr) or (tmp_res!=res_months): 
        print(f'..... Loading itrace dataset for {kyr} ka BP at res {res_months} months')
        itrace_data_global = gutils.get_preprocessed_itrace_data(
            res=itrace_params['res [months]'],
            data_folder=itrace_folder,
            format='df',
            verbose=verbose,
            sim_prefix = itrace_params['Itrace simulation spec']['prefix'],
            sim_suffix =itrace_params['Itrace simulation spec']['suffix'], 
            sim_forcings=itrace_params['Itrace simulation spec']['forcings'],
            sim_kyr= int(itrace_params['Itrace simulation spec']['kyr_filename']),
            sim_num=itrace_params['Itrace simulation spec']['num'],
            sim_model=itrace_params['Itrace simulation spec']['model'],
        ) # df
        itrace_data = itrace_data_global
    
    # sanity check just in case 
    if not isinstance(itrace_data,pd.DataFrame) :
            raise TypeError('data to krige should be a pd df')
    
    # 3. update our temporary values of kyr and res
    tmp_kyr = kyr
    tmp_res = res_months

    # 4. Load the variogram model and params # uncomment if using an aggregated variogram whose path is vario_path
    if vario_subfolders is None:
        with open(vario_path, 'r') as f:
            variogram_dict = json.load(f)
        variogram_model = variogram_dict['model_name'] 
        variogram_parameters = {
            "sill": variogram_dict['sill'], 
            "range": variogram_dict['range'],  
            "nugget": variogram_dict['nugget']
        }
    # =======================================================================================================
    # At this step, we have the external drift df and sisal df at the same temporal resolution and time slice.
    # We can now loop on each of the time steps within the slice to perform kriging.
    # ========================================================================================================
    df_list=[]
    for t_months, itrace_slice in tqdm(itrace_data.groupby("time",observed=True)): 
        
        yrBP_time = utils.get_yrBP_from_itrace_time(t_months,start_year=int(kyr)*1000) #type:ignore
        # OUTPUT FOLDER
        fp_output_yrBP = f"{fp_output}plots_and_metrics/{str(int(yrBP_time))}yrBP/" #type:ignore
        if not os.path.exists(fp_output_yrBP):
            os.makedirs(fp_output_yrBP)

        # LOAD VARIOGRAM PARAMS IF NOT DONE BEFORE :
        if vario_subfolders is not None :
            vario_fp = f"{vario_path}sim{kyr}kyrBP{vario_subfolders}variogram_params_None_t{int(t_months)}.json" #type:ignore
            with open(vario_fp, 'r') as f:
                variogram_dict = json.load(f)
            variogram_model = variogram_dict['model_name'] 
            variogram_parameters = {
                "sill": variogram_dict['sill'], 
                "range": variogram_dict['range'],  
                "nugget": variogram_dict['nugget']
            }

        # EXTRACT THE RIGHT TIME SCLICES
        external_drift_df = itrace_slice.dropna() 
        if any(external_drift_df['lon']>180):
            external_drift_df['lon']=utils.convert_lon_0_360_to_neg180_180(np.asarray(external_drift_df['lon'].values))
        sisal_bin_label = yrBP_time - res_months//12 # bins of sisal are defined as 'N':[N+width yrBP, N yrBP) while itrace is reversed : 'N':(N yrBP, N-width yrBP]...
        cols_to_keep =['site_name','site_id','lon','lat','d18O_measurement','d18Op','d18O_precision']
        data_to_krige = sisal_df[ (sisal_df['time']==sisal_bin_label)].copy()[cols_to_keep].groupby('site_name').mean()

        # KRIGE ON THE ENTIRE SET OF OBSERVATIONS
        df_pred = gutils.ked(
            df_ext_drift=external_drift_df,
            df_to_krige=data_to_krige, 
            lat='lat',
            lon='lon',
            qty ='d18Op',
            variogram_model=variogram_model,
            variogram_parameters=variogram_parameters
        )        
        # CONFIDENCE FLAG
        df_pred['time']=yrBP_time
        df_pred['confidence_flag']=1
        df_pred.loc[df_pred['ss_pred']>8,'confidence_flag']= 2
        df_pred.loc[df_pred['ss_pred']>16,'confidence_flag']=3
        
        # ADD RESULTS TO DF LIST
        df_list.append(df_pred)
        
        # PLOTS OF KED RESULTS
        fig,axes = putils.plot_ked_platecarree_points(
            df_pred=df_pred,
            df_obs=data_to_krige,               
            value_col='d18Op',
            cmap='plasma',
            s_pred=80,
            figsize=(20,4),
            adjust_extent=False
        )
        plt.tight_layout()
        plt.savefig(f'{fp_output_yrBP}map_all_predictions.png',dpi=500)   
        
        fig,axes = putils.plot_ked_platecarree_points(
            df_pred = df_pred[df_pred.confidence_flag < 3],
            df_obs = data_to_krige,               
            value_col = 'd18Op',
            cmap = 'plasma',
            s_pred = 80,
            figsize = (20,4),
            adjust_extent = False
        )
        plt.tight_layout()
        plt.savefig(f'{fp_output_yrBP}map_masked_predictions.png',dpi=500)   
        plt.close('all')

        # CROSS-VALIDATION LOOP 
        if verbose: print('cross-validation...')
        cvresults_list = []
        for site in data_to_krige.index.unique():
            mask = data_to_krige.index==site
            df_pred_val = gutils.ked(
                df_ext_drift=external_drift_df,
                df_to_krige=data_to_krige, 
                lat='lat',
                lon='lon',
                qty ='d18Op',
                variogram_model=variogram_model,
                variogram_parameters=variogram_parameters,
                cv_mask=mask
            ) 
            cvresults_list.append(df_pred_val)         
        cv_df =  pd.concat(cvresults_list).reset_index(drop=False) # so site_name re-appears as a column instead of index
        
        # LOO METRICS COMPUTATION 
        gutils.cv_metrics_and_plots(cv_df,fp_output_yrBP,alpha=alpha)

# Aggregate results and save them 
df = pd.concat(df_list).reset_index(drop=True)
df.rename(columns={'z_pred':'d18Op'},inplace=True)
ds = utils.prepare_ds_of_ked_isoscsape(df)
ds.to_netcdf(f'{fp_output}speleothems_isoscapes_11_to_20kaBP.nc')