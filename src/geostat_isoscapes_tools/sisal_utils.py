import pandas as pd
from pandas import DataFrame
import os 
from xarray import DataArray
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from . import utils
# =============================================================
# Useful functions and constants for handling SISAL data 
# =============================================================

########################################################################################## CONSTANTS
SITES_CHINA = [
    "Kesang cave",
    "Yangkou cave",
    "Hulu cave",
    "Huangye cave",
    "Dayu cave",
    "Dongge cave",
    "Kulishu cave",
    "Suozi cave",
    "Xinglong cave",
    "Furong cave",
    "Xinya cave",
    "Heshang cave",
    "Xiaobailong cave",
    "Sanbao cave",
    "Tianmen",
    "Jiuxian cave",
    "Zhuliuping cave",
    "Yaoba Don cave",
    "Qingtian cave",
    "Yamen cave",
    "Lianhua cave, Hunan",
    "Xiangshui cave",
    "Dark cave",
    "E'mei cave",
    "Nuanhe cave",
    "Shennong cave",
    "Wanxiang cave",
    "Xianglong cave",
    "Baluk cave",
    "Haozhu cave",
    "Lianhua cave, Shanxi",
    "Shenqi cave",
    "Shigao cave",
    "Wuya cave",
    "Zhenzhu cave",
    "Heifeng cave",
    "Huangchao cave",
    "Jiangjun cave",
    "Jinfo cave",
    "Jiulong cave",
    "Linzhu cave",
    "Qujia cave",
    "Xinglong cave",
    "Shijiangjun cave",
    "Shizi cave",
    "Wulu cave",
    "Xiniu cave",
    "Yangzi cave",
    "Zhangjia cave"
]

SITES_CLOSE_TO_CHINA = [
    "Baeg-nyong cave",
    "Tham Doun Mai",
    "Wah Shikhar cave",
    "Mawmluh cave",
    "Timta cave",
    "Sahiya cave",
    "Bittoo cave",
    "Tityana cave",
    "Kalakot cave",
    "Talisman cave",
    "Bir-Uja cave"
]

################################################################################## LOADING & PREPROCESSING
def load_sisal():
    ''' This function reads the SISAL database and return the data of chronology, dating, 
    samples, entities, sites in a dict of dfs. 
    '''
    # read sisalv3.csv files
    cwd = os.getcwd()
    proj_root_dir = utils.get_project_root()
    os.chdir(proj_root_dir+"/data/sisalv3_csv")

    entity_df = pd.read_csv('entity.csv')
    d18O_df   = pd.read_csv("d18O.csv")
    dating_df = pd.read_csv("dating.csv")
    # d13C = pd.read_csv("d13C.csv")
    # MgCa = pd.read_csv("Mg_Ca.csv")

    entity_link_reference_df = pd.read_csv("entity_link_reference.csv")
    original_chronology_df   = pd.read_csv("original_chronology.csv")
    reference_df             = pd.read_csv("reference.csv")
    sample_df                = pd.read_csv("sample.csv")
    sisal_chronology_df      = pd.read_csv("sisal_chronology.csv")
    site_df                  = pd.read_csv("site.csv")
    composite_entity_df      = pd.read_csv("composite_link_entity.csv")
    
    # some columns of the database have a number on the first position of their name which would produce errors in python
    dating_df.rename(columns = {'238U_content':'c238U_content','238U_uncertainty':'c238U_uncertainty',
        '232Th_content':'c232Th_content','c232Th_uncertainty':'c232Th_uncertainty',
        '230Th_content':'c230Th_content','c230Th_uncertainty':'c230Th_uncertainty',
        '230Th_232Th_ratio':'a230Th_232Th_ratio','230Th_232Th_ratio_uncertainty':'a230Th_232Th_ratio_uncertainty',
        '230Th_238U_activity':'a230Th_238U_activity','230Th_238U_activity_uncertainty':'a230Th_238U_activity_uncertainty',
        '234U_238U_activity':'a234U_238U_activity','234U_238U_activity_uncertainty':'a234U_238U_activity_uncertainty'},
        inplace = True)

    os.chdir(cwd)

    return {'original_chronology' : original_chronology_df,
            'sample': sample_df,
            'entity': entity_df,
            'd18O': d18O_df,
            'sisal_chronology': sisal_chronology_df,
            'site': site_df,
            'composite_entity': composite_entity_df,
            'entity_link_reference': entity_link_reference_df,
            'reference': reference_df }

#### DATA PREPROCESSING
def clean_sisal_data(sisal_dict: dict) -> dict : 
    ''' Cleaning the table "sample_df" of the SISAL database to keep only valid samples, based on these criterions :
    1.  remove samples corresponding to superseded entities
    2.  remove samples with mixed mineralogy, keep only calcite or aragonite speleothems
    [ DEPRECATED : 3.  remove samples with no "chrono" age where "chrono" refers to the chronology to use]
    [ DEPRECATED : 4.  remove samples with negative chronology (it is explicitel mentionned in SISAL read me that dates (yr BP) should be positive decimals)]
    3.  set age of samples from original chronology table and sisal chronology table if samples are missing in the original chronology table
    5.  remove samples for which we do not have a d18O measurement
    Inputs :
        - sisal_dict : dict of the sisal database as returned by function load_sisal()
        [DEPRECATED - chrono : str referring to the chronology method of interest. 
                   Samples with no age provided by this chrono methods will be excluded.
                   values : 'sisal-lin-interp','orig-chrono',... #, chrono : str ='lin_interp_age']
    Output :
        - clean_sisal_dict : dict containing ONLY the cleaned dataframes of sites, entities, samples.
    '''
    # Extract the dfs from sisal dict 
    site_df = sisal_dict['site']
    entity_df = sisal_dict['entity']
    sample_df = sisal_dict['sample']
    d18O_df   = sisal_dict['d18O']
    composite_entity_df = sisal_dict['composite_entity']
    original_chrono_df = sisal_dict['original_chronology']
    sisal_chrono_df = sisal_dict['sisal_chronology']
    
    #############################
    # 1) Remove superseded entities and their samples
    entity_df1 = entity_df.loc[entity_df['entity_status']!='superseded']
    sample_df1 = sample_df[sample_df["entity_id"].isin(entity_df1["entity_id"].unique())]
    #############################
    # 2) Exclude samples of mixed mineralogy entities :
    #   Find how many different mineralogies exist in the samples of each entity :
    unique_entities = sample_df1.groupby("entity_id")["mineralogy"].nunique() # dropna=False to count nan as a different value                                                                 
    #   Keep only those with a single mineralogy
    unique_min_sample_df = sample_df1[sample_df1["entity_id"].isin(unique_entities[unique_entities == 1].index)]
    sample_df2 = unique_min_sample_df[unique_min_sample_df["mineralogy"].isin(['calcite', 'aragonite'])]
    #############################
    # 3) Set age of samples either with original chronology or sisal chronology 
    sample_df3 = set_samples_age_and_uncert(sample_df=sample_df2,
                                            original_chronology_df=original_chrono_df,
                                            sisal_chronology_df=sisal_chrono_df)
    #############################
    # 5) Keep only samples existing in d18O_df and having a non nan measurement
    d18O_df1 = d18O_df.dropna(subset='d18O_measurement')
    sample_df4 = sample_df3[(sample_df3["sample_id"].isin(d18O_df1["sample_id"]))]

    ##############################
    # 6) Remove composite entities
    entity_df2 = entity_df1[~ entity_df1['entity_id'].isin(composite_entity_df['composite_entity_id'])]
    sample_df5 = sample_df4[sample_df4['entity_id'].isin(entity_df2['entity_id'])]

    ##############################
    # Final, clean, concording, dataframes
    final_entity_df = entity_df2[entity_df2["entity_id"].isin(sample_df5["entity_id"])]
    final_d18O_df = d18O_df[d18O_df["sample_id"].isin(sample_df5["sample_id"])]
    final_site_df = site_df[site_df["site_id"].isin(final_entity_df["site_id"])]

    return {'sample': sample_df5,
            'entity': final_entity_df,
            'd18O': final_d18O_df,
            'site': final_site_df,
            }

def merge_sisal_df_with_columns(site_df   : DataFrame,
                                entity_df : DataFrame,
                                sample_df : DataFrame,
                                d18O_df   : DataFrame,
                                col_entity: list,
                                col_site  : list,
                                col_sample: list,
                                col_d18O  : list)-> DataFrame :
    ''' This function merges the SISAL dataframes site_df, entity_df, sample_df, chrono_df and d18O_df into a single DataFrame, 
    keeping only the columns specified in lists col_entity, col_site, etc, from each dataframe.
    /!/ Columns serving as index (sample_id, entity_id, etc) that are used to link the different dataframes (herited from sql tables) 
    are automatically conserved and should not be specified in the lists. 
    Inputs : 
        - site_df   : DataFrame of the sisal database containing sites info
        - entity_df : DataFrame of the sisal database containing entities info
        - sample_df : DataFrame of the sisal database containing samples info
        - d18O_df   : DataFrame of the sisal database containing d18O info.
    Outputs :
        - merged_df : DataFrame containing the columns specified in list arguments and the columns of indices "sample_id","entity_id" and "site_id".
    '''

    entities1 = entity_df[ col_entity+["site_id","entity_id"]].merge(site_df[ col_site+["site_id"] ],on="site_id",how='left')
    sample1 = sample_df[ col_sample+["entity_id","sample_id"] ].merge(entities1, on="entity_id",how='left')
    # sample2 = sample1.merge(chrono_df[ col_chrono+["sample_id"] ], on="sample_id",how='left')
    merged_df = sample1.merge(d18O_df[ col_d18O+["sample_id"] ], on="sample_id", how='left')

    return merged_df

def get_basic_cleaned_merged_sisal_data(verbose=True)-> DataFrame:
    ''' Loads the SISAL database, apply basic cleaning (see documentation of function clean_sisal_data above),
    and merges the different tables of the database into one dataframe 
    (keeping only the commonly needed columns TODO:add flexibilizy in columns choice)
    '''
    # Load SISAL data
    if verbose : print('-> loading SISAL database')
    sisal_dict = load_sisal()

    # Clean the data
    if verbose : print('   cleaning samples')
    clean_dict = clean_sisal_data(sisal_dict)
    site_df_clean = clean_dict['site']
    entity_df_clean = clean_dict['entity']
    d18O_df_clean = clean_dict['d18O']
    sample_df_clean = clean_dict['sample']

    # Merge in one df
    col_site = ['site_name','longitude','latitude']
    col_entity = []
    col_sample = ["mineralogy","age","age_method","age_uncert_pos","age_uncert_neg"]
    col_d18O = ['d18O_measurement','d18O_precision']

    merged_data = merge_sisal_df_with_columns(
        site_df= site_df_clean,
        entity_df  = entity_df_clean,
        sample_df  = sample_df_clean,
        d18O_df    = d18O_df_clean,
        col_entity = col_entity,
        col_d18O   = col_d18O,
        col_site   = col_site,
        col_sample = col_sample
        )
    if verbose : print('loading and cleaning done.')
    return merged_data

def set_samples_age_and_uncert(sample_df : pd.DataFrame,
                               original_chronology_df : pd.DataFrame,
                               sisal_chronology_df: pd.DataFrame):
    # put all ages in the samples dataframe
    sample_df_interp_age  = pd.merge(sample_df,original_chronology_df[['sample_id','interp_age','interp_age_uncert_pos','interp_age_uncert_neg']],how='left')
    age_models = ['lin_interp_age','lin_reg_age','Bchron_age','Bacon_age','OxCal_age','copRa_age','StalAge_age']
    cols = age_models + [c+'_uncert_pos' for c in age_models] + [c+'_uncert_neg' for c in age_models] + ['sample_id']
    out_df = pd.merge(sample_df_interp_age,sisal_chronology_df[cols],how='left')

    # set the age value to the interp_age value, if it exists
    out_df['age']=np.nan
    out_df['age_uncert_pos']=np.nan
    out_df['age_uncert_neg']=np.nan
    out_df['age_method'] = ''
    out_df.loc[~out_df['interp_age'].isna(),'age']=out_df['interp_age']
    out_df.loc[~out_df['interp_age'].isna(),'age_method']='interp_age'
    out_df.loc[~out_df['interp_age'].isna(),'age_uncert_neg']=out_df['interp_age_uncert_neg']
    out_df.loc[~out_df['interp_age'].isna(),'age_uncert_pos']=out_df['interp_age_uncert_pos']

    # for samples with no interp_age, see which methods are available
    for method in age_models : 
        out_df.loc[(out_df['interp_age'].isna())&(~out_df[method].isna()),'age_method']+= method #type:ignore
    # Sometimes it is easy : only one age model is available :
    for method in age_models :
        out_df.loc[out_df['age_method']==method,'age'] = out_df[method]
        out_df.loc[out_df['age_method']==method,'age_uncert_pos'] = out_df[f"{method}_uncert_pos"]
        out_df.loc[out_df['age_method']==method,'age_uncert_neg'] = out_df[f"{method}_uncert_neg"]

    # .. but when we have multiple age models choices : we do not prefer one age model to another, so we can aggregate the models with inverse variance weighted mean
    mask_mult_methods = ~((out_df['age_method'].isin(cols))|(out_df['age_method']=='interp_age'))
    list_multiple_methods = out_df.loc[mask_mult_methods,'age_method'].unique()    

    for l in list_multiple_methods :
        # get the list of methods from the long string
        ms = [m + '_age' for m in l.split('_age')]
        ms.pop()
        mask_l_method = out_df['age_method']==l # mask = selects rows with methods ms (l)
        df_l = out_df[mask_l_method].copy() # df of rows sharing same methods ms (l)
        # compute weighted mean
        for m in ms : # for each method m among methods ms (l)
            # prepare columns w, sigma
            df_l[f'{m}_w']=np.nan
            df_l[f'{m}_sigma']=np.nan
            up  = df_l[f"{m}_uncert_pos"]
            un  = df_l[f"{m}_uncert_neg"]
            # compute sigma
            nan_mask = ~up.isna() & ~un.isna()
            df_l.loc[nan_mask,f'{m}_sigma'] =  (up[nan_mask] + un[nan_mask]) / 2.0
            # compute w : 
            nonzero_mask = df_l[f'{m}_sigma']!=0
            df_l.loc[nan_mask & nonzero_mask,f'{m}_w'] = w = 1.0 / (df_l.loc[nan_mask & nonzero_mask,f'{m}_sigma']**2)
            # df_l.loc[(~nan_mask)|(~nonzero_mask),f'{m}_w'] = np.nan   # CHOICE : nan weight if missing or 0 uncertainty : we do not want to use this value
        
        # Normalized weighted mean
        ages = np.array(df_l[ms], float)
        weights = np.array(df_l[[m+"_w" for m in ms]], float)

        df_l['age'] = np.nansum(weights * ages,axis=1) / np.nansum(weights,axis=1)
        # The uncertainty is the within uncertainety + methods disagreement variance
        within_sigma2 = 1 / np.nansum(weights,axis=1) # combination of the methods uncertainties
        between_sigma2 = np.nanvar(ages, axis=1) # uncertainty due to the disagreement of methods
        df_l['age_uncert_pos'] = np.sqrt(within_sigma2 + between_sigma2)
        df_l['age_uncert_neg'] = df_l['age_uncert_pos']

        # put the ages and uncert. in the out_df columns
        out_df.loc[mask_l_method,'age'] = df_l['age']
        out_df.loc[mask_l_method,'age_uncert_neg'] = df_l['age_uncert_neg']
        out_df.loc[mask_l_method,'age_uncert_pos'] = df_l['age_uncert_pos']
        
    return out_df

def convert_calcite_to_drip_water(calcite_df : DataFrame) -> DataFrame :
    ''' This functions converts calcite d18O (V-PDB standards) values to their drip water equivalent (V-SMOW standard), 
    using the mineralogy and temperature associated to each sample to convert.
    Input : 
        - caclite_df : DataFrame including columns d18O_meansurement (float64),T (float64) in K, mineralogy (str)
    Outout :
        - converted_df : copy of calcite_df with an extra column d180p_VSMOW containing the drip water d18O V-SMOW values.
    '''
    # from literature (Comas-Bru 2019)
    conversion_cst = {'calcite'  :[16.1,24.6],
                      'aragonite':[18.34,31.954]
                      }
    converted_data = calcite_df.copy()
    converted_data['d18Op_VSMOW_exactconv'] = np.nan
    converted_data['d18Op_VSMOW_linearized'] = np.nan

    for mineralogy in ['calcite','aragonite']:
        mask = converted_data["mineralogy"]==mineralogy
        d18O = converted_data.loc[mask,'d18O_measurement']
        T = converted_data.loc[mask,'T']
        c0,c1 = conversion_cst[mineralogy]
        
        # convert PDB to VSMOW standard
        converted_data.loc[mask,'d18Oc_VSMOW'] = 1.03092*d18O + 30.92 
        
        # use Tremaine equation to define the calcite-to-water fractionation factor
        alpha =  np.exp((( c0*1000/T.astype(float) - c1) / 1000))
        
        # use the definition of a fractionation factor and the definition of the delta 18O to convert calcite delta 18O to its water equivalent
        converted_data.loc[mask,'d18Op_VSMOW_exactconv'] = (1000 + converted_data.loc[mask,'d18Oc_VSMOW']) /alpha - 1000 
        converted_data.loc[mask,'d18Op_VSMOW_linearized'] = 1.03092*d18O + 30.92 - ( c0*1000/T.astype(float) - c1 ) # orig. from Comas-Bru 2019 : wrong ? 
        # print(f"   for mineralogy {mineralogy}, the conversion failed for {converted_data.loc[mask,'d18Op_VSMOW'].isna().sum()} samples.")
    return converted_data

def retrieve_T_RegularGridInterp(data_df : DataFrame, 
                                 temp_xda : DataArray, 
                                 method : str = 'linear',
                                 data_lon_col='longitude',
                                 data_lat_col='latitude',
                                 temp_lon_col='lon',
                                 temp_lat_col='lat'
                                 ) -> DataFrame :
    ''' Retrieve the temperature of each sample of data_df by interpolating temp_xda values at data_df points using scipy RegularGridInterpolator
    Inputs :
        - data_df : DataFrame containing columns 'latitude', ' longitude', chrono. The chrono column should contain **positive** ages in yrs BP
        - temp_xda : DataArray containing a global temperature dataset with dimensions 'lat','lon','time'. 
                    
        - method : str of the name of the method to use to interpolate the temperature data points. Supported : "linear", "nearest", "slinear", "cubic", "quintic" and "pchip".
    Output : 
        - data_df : with an exra column 'T_interp' containing the temperature associated to each sample row.
    '''
    # set up the interpolator
    temp_lats  = temp_xda[temp_lat_col].values.copy()
    temp_lon   = temp_xda[temp_lon_col].values.copy()
    temp_times = temp_xda['time'].values.copy()
    temp_values = temp_xda.values

    # check that all lons and lats are defined in the right convention 
    if any(temp_lats>90):
        temp_lats = utils.convert_lat_0_180_to_neg90_90(temp_lats)
        # sort latitudes (ow it breaks RegularGridInterpolator)
        lat_idx = np.argsort(temp_lats)
        temp_lats = temp_lats[lat_idx]
        temp_values = temp_values[..., lat_idx]
    
    if any(temp_lon>180):
        temp_lon = utils.convert_lon_0_360_to_neg180_180(temp_lon)
        # sort longitudes (ow it breaks RegularGridInterpolator)
        lon_idx = np.argsort(temp_lon)
        temp_lon = temp_lon[lon_idx]
        temp_values = temp_values[..., lon_idx]

    interp_func = RegularGridInterpolator(
        points = (temp_times, temp_lats, temp_lon),
        values = temp_values,
        bounds_error = False,
        fill_value = np.nan,
        method = method
    )

    # set up sample points for interpolation
    sample_times = np.asarray(data_df['age'].values)
    sample_lats = np.asarray(data_df[data_lat_col].values)
    sample_lons = np.asarray(data_df[data_lon_col].values)

    # check that all lons and lats are defined in the right convention 
    if any(sample_lats> 90):
        sample_lats = utils.convert_lat_0_180_to_neg90_90(sample_lats)
    if any(sample_lons> 180):
        sample_lons = utils.convert_lon_0_360_to_neg180_180(sample_lons)

    points = np.column_stack([sample_times, sample_lats, sample_lons]) # type: ignore

    # interpolate
    data_df[f'T_{method}'] = interp_func(points)
    
    return data_df

def retrieve_temperature_and_convert_speleothem_d18O(data_df : DataFrame, temp_xda : DataArray, method : str = 'linear', verbose: bool = True)-> DataFrame :
    ''' 1) retrieves temperature of data_df samples based on the temp_xda datarray provided (interpolation with specified method + NN as backup) 
        2) convert speleothem PDB d18O into precipitation VSMOW d18O using Tremaine equation.
        Be careful : the convention for the dates should be the same for both data_df and temp_xda (for instance, time dim can be years before present, so that 12 means 12 years BP.)
    '''
    if verbose : print("-> converting speleothem data")
    # 1. Temperature retrieval 
    data_to_convert = retrieve_T_RegularGridInterp(data_df = data_df,temp_xda = temp_xda, method = 'linear')
    #    Mask locs and times for which this method failed
    mask_nan = data_to_convert['T_linear'].isna()
    #    Apply the NN method for these points, if any
    if mask_nan.sum() :
        if verbose : print(f'   {method} interpolation failed for {mask_nan.sum()} samples, trying to fill missing T with nearest neighbour method')
        data_to_convert['T_nearest'] = pd.NA
        data_to_convert.loc[mask_nan,'T_nearest'] = retrieve_T_RegularGridInterp(data_df=data_df[mask_nan].copy(),
                                                                                 temp_xda=temp_xda,
                                                                                 method='nearest')['T_nearest']
    #    Gather T in a T column
    data_to_convert.loc[ mask_nan,'T'] = data_to_convert.loc[mask_nan,'T_nearest']
    data_to_convert.loc[~mask_nan,'T'] = data_to_convert.loc[~mask_nan,'T_linear']
    mask_nans_final = data_to_convert['T'].isna()
    if verbose : print(f'   after {method} interpolation and nearest neighbour backup, still no temperature for', mask_nans_final.sum(),'samples.')
    if verbose : print("   temperature retrieval finished, starting conversion")
    
    # 2. Conversion 
    converted_data = convert_calcite_to_drip_water(data_to_convert)
    if verbose : print("conversion done.")

    return converted_data

def retrieve_continent_from_lat_lon(df_orig : pd.DataFrame,lat_col :str ='latitude',lon_col :str ='longitude') -> pd.DataFrame :
    ''' Rough definition of continents with "rectangles"
    '''
    df = df_orig.copy()
    # Define regions
    df['continent'] = ''
    #middle east apart form africa :
    df.loc[(df[lat_col]>= -35)&(df[lat_col]<= 37)&(df[lon_col]>= -20)&(df[lon_col]<= 52),'continent']='Africa'
    df.loc[(df[lat_col]<= -60),'continent']='Antarctica'
    df.loc[(df[lat_col]>= 5)&(df[lat_col]<= 81)&(df[lon_col]>= 26)&(df[lon_col]<= 180),'continent']='Asia'
    df.loc[(df[lat_col]>= 35)&(df[lat_col]<= 72)&(df[lon_col]>= -25)&(df[lon_col]<= 50),'continent']='Europe'
    df.loc[(df[lat_col]>= 5)&(df[lat_col]<= 83)&((df[lon_col]>= -170)&(df[lon_col]<= -50)),'continent']='North America'
    df.loc[(df[lat_col]>= -50)&(df[lat_col]<= 0)&((df[lon_col]>= 110)|(df[lon_col]<= 180)),'continent']='Oceania'
    df.loc[(df[lat_col]>= -60)&(df[lat_col]<= 15)&(df[lon_col]>= -90)&(df[lon_col]<= -30),'continent']='South America'
    df.loc[(df['continent']=='') & (df[lat_col]>= 0) & (df[lat_col]<= 15) & (df[lon_col]>= 110) & (df[lon_col]<= 120),'continent']='Indonesia' # Indonesie, transcontinental
    df.loc[(df['continent']=='') & (df[lat_col]>= 75) & (df[lat_col]<= 85) & (df[lon_col]>= -30) & (df[lon_col]<= -15),'continent']='North America' # Greenland is on North America plate
    df.loc[(df[lat_col]>= 12)&(df[lat_col]<= 42)&(df[lon_col]>= 32)&(df[lon_col]<= 60),'continent']='Middle East' # Middle East

    continents = df['continent'].unique()
    # print('-> Continents found in the site df :',continents)
    return df

