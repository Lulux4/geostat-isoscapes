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

exp_name  = 'runs_wip/2026-06-22/vario_n10/alpha10_percent/crop_lat_2/' # name of expermiment 

# Caves to exclude (sensitivity analysis...)
# caves_to_exclude = ['Pacupahuain cave']
caves_to_exclude = []

# Or all varios :
vario_path = f'{utils.get_project_root()}/output/variograms/2026-06-12-PHIS/itrace/' #vario_res200y_maxlag10e6_nlags15_latlatquadeleD_mask2000.json
vario_subfolders = '/res200/maxlag10000000nlags10/trend_multiple_linear_lat_latquad_ele_D/sisal_mask_2000km/'

scaling_factor = 1.0 #0.90 
alpha = 0.1 # max acceptable type I error rate for the Kleijnen test

temperature_ds_name = 'itrace' # 1) 'itrace' for 20-11 ka BP, or 2) 'krapp' for 0-800ka BP with res 1000 years
temperature_ds_path = '/media/luluxette/T7_Shield/pdm/iTrace/'

itrace_folder = '/media/luluxette/T7_Shield/pdm/iTrace/' # abs path to itrace data 

with open(f'{utils.get_project_root()}/data/iTrace_simulations_dict.json', 'r') as f:
    itrace_sims = json.load(f)
# Keep only the simulations of interest (the keys represent 1000 yr BP slices)
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

res_list = [200] # min resolution is 1 year

verbose = False

# Prepare the iterations 
iters = itertools.product(res_list,sims.keys())
tmp_kyr = None
tmp_res = None
df_list = []

for res,kyr in iters :
    print('====================================================================================================')
    print(f'Time={kyr} ka BP, resolution={res} years')
    print('====================================================================================================')
    # ======================= CONFIGURATION =======================
    params = {
        'kyr' : kyr,
        'trend_before_mask': True,
        'tolerance' : 22.5,
        'vario_path' : vario_path,
        'res [years]': res,
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
    fp_output = f"{utils.get_project_root()}/output/kriging/{exp_name}/res{itrace_params['res [years]']}/"
    if not os.path.exists(fp_output):
        os.makedirs(fp_output)
    
    # ================= DATASETS LOADING ====================================================
    # 1. If res changed, we reload sisal global data
    if (tmp_res!=res):
        print(f'..... Loading sisal dataset, resolution {res}')
        sisal_df = gutils.get_sisal_data_for_kriging(
            res=int(res),
            temp_ds_name=temperature_ds_name,
            temp_ds_path=temperature_ds_path,
            verbose=verbose
        )
        if not isinstance(sisal_df,pd.DataFrame) :
            raise TypeError('data to krige should be a pd df')
        sisal_df = sisal_df.rename(columns={'binned_age':'time','d18Op_VSMOW_exactconv':'d18Op'})
        sisal_df['d18Op'] = np.array(sisal_df.d18Op,dtype=np.float64)
        
        # Remove Babylon cave as it is located at the exact same coordinates as Hollywood cave : pb for the kriging system
        sisal_df  = sisal_df[(sisal_df.site_name != 'Babylon cave')]
        
        # remove other caves as specified in parameters
        sisal_df  = sisal_df[~sisal_df.site_name.isin(caves_to_exclude)]

    # 2. Reload itrace whenever the time slice changes -- bc separate files --
    if (tmp_kyr is None) or (kyr!=tmp_kyr) or (tmp_res!=res): 
        print(f'..... Loading itrace dataset for {kyr} ka BP at res {res} years')
        itrace_data_global = gutils.get_preprocessed_itrace_data(
            true_kyr=int(kyr),
            res=itrace_params['res [years]'],
            data_folder=itrace_folder,
            format='df',
            amount_weighted=True,
            Ele=True,
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
    tmp_res = res

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
    for t, itrace_slice in tqdm(itrace_data.groupby("time",observed=True)): 
        
        # OUTPUT FOLDER
        fp_output_yrBP = f"{fp_output}plots_and_metrics/{str(int(t))}yrBP/" #type:ignore
        if not os.path.exists(fp_output_yrBP):
            os.makedirs(fp_output_yrBP)

        # LOAD VARIOGRAM PARAMS IF NOT DONE BEFORE :
        if vario_subfolders is not None :
            vario_fp = f"{vario_path}sim{kyr}kyrBP{vario_subfolders}variogram_params_None_t{int(t)}.json" #type:ignore
            with open(vario_fp, 'r') as f:
                variogram_dict = json.load(f)
            list_names = [param_name.split('_')[1]  for param_name in variogram_dict.keys() if (('range_' in param_name) or ('sill_' in param_name))] 
            variogram_model = list(set(list_names))
            if len(variogram_model)>1 : 
                raise NotImplementedError('composite models are currently not supported')
            
            variogram_model = variogram_model[0]
            if 'nugget' not in list(variogram_dict.keys()): variogram_dict['nugget']=0.0
            
            variogram_parameters = {
                "sill": variogram_dict[f'sill_{variogram_model}']*scaling_factor, 
                "range": variogram_dict[f'range_{variogram_model}'],  
                "nugget": variogram_dict['nugget']
            }

        # EXTRACT THE RIGHT TIME SCLICES
        external_drift_df = itrace_slice.dropna() 
        if any(external_drift_df['lon']>180):
            external_drift_df['lon']=utils.convert_lon_0_360_to_neg180_180(np.asarray(external_drift_df['lon'].values))
        sisal_bin_label = t # deprecated-> bins of sisal are defined as 'N':[N+width yrBP, N yrBP) while itrace is reversed : 'N':(N yrBP, N-width yrBP]...
        cols_to_keep =['site_name','site_id','lon','lat','d18O_measurement','d18Op','d18O_precision','ele']
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
        df_pred['time']=t
        df_pred = df_pred[df_pred.lat.between(-62.9,73)]#(-54.2,64.9)] 
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
        plt.close('all')

# Aggregate results and save them 
df = pd.concat(df_list).reset_index(drop=True)
df.rename(columns={'z_pred':'d18Op'},inplace=True)
ds = utils.prepare_ds_of_ked_isoscsape(df)
ds.to_netcdf(f'{fp_output}speleothems_isoscapes_11_to_20kaBP.nc')