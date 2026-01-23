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

###############################################
# Define parameters to apply successively
###############################################
vario_path = f'{utils.get_project_root()}/output/variograms/itrace/sim12kyrBP/res2400/maxlag12000000nlags20/trend_multiple_linear_latabs_latReLU_ele_D/sisal_mask/variogram_params_None.json'

ss_threshold = 16

sims = { # simulation spec for itrace dataset
    # '1':{}, # dummy entry for sisal only, to keep all times
    '12':{'num':'05','suffix':'800001-899912'},
    '16':{'num':'01','suffix':'400001-499912'},
    '20':{'num':'01','suffix':'100001-199912'}
    }
res_months_list = [12*200] # min resolution is 12 months if using sisal, 1 month if using itrace
verbose = False
# Prepare the iterations 
iters = itertools.product(res_months_list,sims.keys())

tmp_kyr = None
tmp_res = None

for res_months,kyr in iters :
    print('====================================================================================================')
    print(f'Time={kyr} kyr BP, resolution={res_months} months')
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
    itrace_params['Itrace simulation spec'] = { 'kyr': kyr,                   
                                            'num': sims[kyr]['num'],                
                                            'model':'clm2.h0',
                                            'forcings':'ice_ghg_orb_wtr',   
                                            'suffix': sims[kyr]['suffix'], 
                                            'prefix':'b.e13.Bi1850C5.f19_g16'}
    fp_output = f"{utils.get_project_root()}/output/kriging/res{itrace_params['res [months]']}/"
    if not os.path.exists(fp_output):
        os.makedirs(fp_output)
    
    # ================= DATA LOADING =====================
    # set a change "tracker" to avoid repeating if statements
    sisal_change = False
    # 1. If the resolution changed, we must reload sisal global data
    if (tmp_res!=res_months):
        print('----- New sisal resolution : loading sisal global df')
        sisal_df_global = gutils.get_sisal_data_for_kriging(res=int(res_months/12),verbose=verbose)
        sisal_df = sisal_df_global
        sisal_change = True    
    # sanity check in case user put wrong parameters and the output was a xr dataarray or None
    if not isinstance(sisal_df,pd.DataFrame) :
            raise TypeError('data to krige should be a pd df')
    
    # 2. If the temporal slice (kyr) is not the same as previous iteration, 
    #    we just need to take the right slice of sisal data (no reload)
    if ((tmp_kyr is None)|((kyr!=tmp_kyr))|(tmp_res!=res_months)) | sisal_change :
        print('----- New sisal upper-level params region or res was set, or new temporal slice param detected : taking the new slice from sisal df')
        if kyr!='all':
            sisal_df_valid = sisal_df[(sisal_df['binned_age']>=(int(kyr)-1)*1000)&(sisal_df['binned_age']<int(kyr)*1000)].rename(columns={'binned_age':'time','d18Op_VSMOW_exactconv':'d18Op'}).copy()
        else :
            sisal_df_valid = sisal_df.rename(columns={'binned_age':'time','d18Op_VSMOW_exactconv':'d18Op'}).copy()

    # 3. For itrace, we need to reload in any case since we cannot load several slices at the smae time
    if (tmp_kyr is None) or (kyr!=tmp_kyr) or (tmp_res!=res_months): 
        print('----- New itrace params res or kyr : loading appropriate data')
        itrace_data_global = gutils.get_preprocessed_itrace_data(
            res=itrace_params['res [months]'],
            format='df',
            verbose=verbose,
            sim_prefix = itrace_params['Itrace simulation spec']['prefix'],
            sim_suffix =itrace_params['Itrace simulation spec']['suffix'], 
            sim_forcings=itrace_params['Itrace simulation spec']['forcings'],
            sim_kyr= int(itrace_params['Itrace simulation spec']['kyr']),
            sim_num=itrace_params['Itrace simulation spec']['num'],
            sim_model=itrace_params['Itrace simulation spec']['model'],
        ) # df
        itrace_data = itrace_data_global
    
    # sanity check just in case 
    if not isinstance(itrace_data,pd.DataFrame) :
            raise TypeError('data to krige should be a pd df')
    
    # 5. update our temporary values of kyr and res
    tmp_kyr = kyr
    tmp_res = res_months

    # 6. Load the variogram 
    with open(vario_path, 'r') as f:
        variogram_dict = json.load(f)
    variogram_model = variogram_dict['model_name'] 
    variogram_parameters = {
        "sill": np.exp(variogram_dict['sill_ln']), 
        "range": variogram_dict['range'],  
        "nugget": np.exp(variogram_dict['nugget_ln'])+1
    }
    # =======================================================================================================
    # At this step, we have the external drift df and sisal df at the same temporal resolution and time slice.
    # We can now loop on each of the time steps within the slice to perform kriging.
    # ========================================================================================================
    df_list=[]
    for t_months, itrace_timestep in tqdm(itrace_data.groupby("time",observed=True)): 
        yrBP_time = utils.get_yrBP_from_itrace_time(t_months,start_year=int(kyr)*1000) #type:ignore
        print(f'> t={int(yrBP_time)}yrBP')        
        # create subfolder for storing metrics and plots outputs cleanly
        fp_output_yrBP = f"{fp_output}plots_and_metrics/{str(int(yrBP_time))}yrBP/"
        if not os.path.exists(fp_output_yrBP):
            os.makedirs(fp_output_yrBP)
        # Extract the time step form itrace and sisal data
        external_drift_df = itrace_timestep.dropna() 
        if any(external_drift_df['lon']>180):
            external_drift_df['lon']=utils.convert_lon_0_360_to_neg180_180(np.asarray(external_drift_df['lon'].values))
        # plot_utils.plot_global_map(external_drift_df,'external drift','d18Op',unit='‰ VSMOW', lat_col='lat',lon_col='lon') 
        sisal_bin_label = yrBP_time - res_months//12 # bins of sisal are defined as 'N':[N+width yrBP, N yrBP) while itrace is reversed : 'N':(N yrBP, N-width yrBP]...
        cols_to_keep =['site_name','site_id','lon','lat','d18O_measurement','d18Op','d18O_precision']
        data_to_krige = sisal_df_valid[ (sisal_df_valid['time']==sisal_bin_label)].copy()[cols_to_keep].groupby('site_name').mean()
        # plot_utils.plot_global_map(data_to_krige,f'SISAL data to krige - {yrBP_time} kyr BP',unit='‰ VSMOW',quantity_col='d18Op',lat_col='lat',lon_col='lon') 

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
        # mask predictions according to a kriging variance threshold
        df_pred['time']=yrBP_time
        df_pred_masked = df_pred[df_pred['ss_pred']<ss_threshold]
        # add the df into a big df
        df_list.append(df_pred_masked)
        # plot
        fig,axes = putils.plot_ked_platecarree_points(
            df_pred=df_pred_masked,
            df_obs=data_to_krige,               
            value_col='d18Op',
            cmap='plasma',
            s_pred=80,
            figsize=(20,4),
            adjust_extent=False
        )
        plt.tight_layout()
        plt.savefig(f'{fp_output_yrBP}map_all_predictions.png',dpi=500)   
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
        # compute metrics on this df : by row and global 
        gutils.cv_metrics_and_plots(cv_df,fp_output_yrBP)

    # aggregate results and save them 
    df = pd.concat(df_list).reset_index(drop=True)
    df.rename(columns={'z_pred':'d18Op'},inplace=True)
    # df.to_csv(f'{fp_output}df_{kyr}kyrBP.csv')
    ds = utils.prepare_ds_of_ked_isoscsape(df)
    ds.to_netcdf(f'{fp_output}ds_{kyr}kyrBP.nc')