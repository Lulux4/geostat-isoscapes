import pandas as pd
from pandas import DataFrame
import os 
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from xarray import DataArray
from scipy.interpolate import RegularGridInterpolator
import numpy as np

############################################################################################### CONSTANTS
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
    os.chdir("../data/sisalv3_csv")

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

#### DATA PROCESSING
def clean_sisal_data(sisal_dict: dict, chrono : str ='lin_interp_age') -> dict :
    ''' Cleaning the table "sample_df" of the SISAL database to keep only valid samples, based on these criterions :
    1.  remove samples corresponding to superseded entities
    2.  remove samples with mixed mineralogy, keep only calcite or aragonite speleothems
    3.  remove samples with no "chrono" age where "chrono" refers to the chronology to use
    4.  remove samples with negative chronology (it is explicitel mentionned in SISAL read me that dates (yr BP) should be positive decimals)
    5.  remove samples for which we do not have a d18O measurement
    Inputs :
        - sisal_dict : dict of the sisal database as returned by function load_sisal()
        - chrono : str referring to the chronology method of interest. 
                   Samples with no age provided by this chrono methods will be excluded.
                   values : 'sisal-lin-interp','orig-chrono',... 
    Output :
        - clean_sisal_dict : dict containing ONLY the cleaned dataframes of sites, entities, samples, and chronology.
    '''
    # Extract the dfs from sisal dict 
    site_df = sisal_dict['site']
    entity_df = sisal_dict['entity']
    sample_df = sisal_dict['sample']
    d18O_df   = sisal_dict['d18O']
    if chrono == 'interp_age':
        ckey = 'original_chronology'
    else :
        ckey = 'sisal_chronology'
    chronology_df = sisal_dict[ckey]
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
    # 3) Keep only samples associated with a ( 4. positive) chronology age
    # print(chronology_df.loc[(chronology_df["sample_id"].isin(sample_df2["sample_id"]) ) 
    #                            & ~( pd.isna(chronology_df[chrono])) &  (chronology_df[chrono]<=-75),
    #                            chrono])
    chrono_df1 = chronology_df[   (chronology_df["sample_id"].isin(sample_df2["sample_id"]) ) 
                               & ~( pd.isna(chronology_df[chrono]))
                               &  (chronology_df[chrono]>=0)
                               ]
    sample_df3 = sample_df2[sample_df2["sample_id"].isin(chrono_df1["sample_id"])]
    #############################
    # 5) Keep only samples existing in d18O_df and having a non nan measurement
    d18O_df1 = d18O_df.dropna(subset='d18O_measurement')
    sample_df4 = sample_df3[(sample_df3["sample_id"].isin(d18O_df1["sample_id"]))]


    final_entity_df = entity_df[entity_df["entity_id"].isin(sample_df4["entity_id"])]
    final_chrono_df = chronology_df[chronology_df["sample_id"].isin(sample_df4["sample_id"])]
    final_d18O_df = d18O_df[d18O_df["sample_id"].isin(sample_df4["sample_id"])]
    final_site_df = site_df[site_df["site_id"].isin(final_entity_df["site_id"])]
    
    return {ckey : final_chrono_df,
            'sample': sample_df4,
            'entity': final_entity_df,
            'd18O': final_d18O_df,
            'site': final_site_df,
            }

def merge_sisal_df_with_columns(site_df   : DataFrame,
                                entity_df : DataFrame,
                                sample_df : DataFrame,
                                chrono_df : DataFrame,
                                d18O_df   : DataFrame,
                                col_entity: list,
                                col_site  : list,
                                col_sample: list,
                                col_chrono: list,
                                col_d18O  : list)-> DataFrame :
    ''' This function merges the SISAL dataframes site_df, entity_df, sample_df, chrono_df and d18O_df into a single DataFrame, 
    keeping only the columns specified in lists col_entity, col_site, etc, from each dataframe.
    /!/ Columns serving as index (sample_id, entity_id, etc) that are used to link the different dataframes (herited from sql tables) 
    are automatically conserved and should not be specified in the lists. 
    Inputs : 
        - site_df   : DataFrame of the sisal database containing sites info
        - entity_df : DataFrame of the sisal database containing entities info
        - sample_df : DataFrame of the sisal database containing samples info
        - chrono_df : DataFrame of the sisal database containing chronology info. Can be the df of sisal_chronology or original_chronology.
        - d18O_df   : DataFrame of the sisal database containing d18O info.
    Outputs :
        - merged_df : DataFrame containing the columns specified in list arguments and the columns of indices "sample_id","entity_id" and "site_id".
    '''

    entities1 = entity_df[ col_entity+["site_id","entity_id"]].merge(site_df[ col_site+["site_id"] ],on="site_id",how='left')
    sample1 = sample_df[ col_sample+["entity_id","sample_id"] ].merge(entities1, on="entity_id",how='left')
    sample2 = sample1.merge(chrono_df[ col_chrono+["sample_id"] ], on="sample_id",how='left')
    merged_df = sample2.merge(d18O_df[ col_d18O+["sample_id"] ], on="sample_id", how='left')

    return merged_df

def get_basic_cleaned_merged_sisal_data(chrono : str ='interp_age')-> DataFrame:
    ''' Loads the SISAL database, apply basic cleaning (see documentation of function clean_sisal_data above),
    and merges the different tables of the database into one dataframe 
    (keeping only the commonly needed columns TODO:add flexibilizy in columns choice)
    '''
    # Load SISAL data
    print('-> loading database')
    sisal_dict = load_sisal()

    # Clean the data
    print('   cleaning samples')
    if chrono == 'interp_age':
        chronology_df_ref = 'original_chronology'
    else : 
        chronology_df_ref = 'sisal_chronology'
    clean_dict = clean_sisal_data(sisal_dict,chrono=chrono)
    site_df_clean = clean_dict['site']
    entity_df_clean = clean_dict['entity']
    d18O_df_clean = clean_dict['d18O']
    sample_df_clean = clean_dict['sample']
    chrono_df_clean = clean_dict[chronology_df_ref]

    # Merge in one df
    col_site = ['site_name','longitude','latitude']
    col_entity = []
    col_chrono = [chrono]
    col_sample = ["mineralogy"]
    col_d18O = ['d18O_measurement','d18O_precision']

    merged_data = merge_sisal_df_with_columns(
        site_df= site_df_clean,
        entity_df  = entity_df_clean,
        sample_df  = sample_df_clean,
        chrono_df  = chrono_df_clean,
        d18O_df    = d18O_df_clean,
        col_entity = col_entity,
        col_chrono = col_chrono,
        col_d18O   = col_d18O,
        col_site   = col_site,
        col_sample = col_sample
        )
    print('loading and cleaning done.')
    return merged_data

def convert_calcite_to_drip_water(calcite_df : DataFrame) -> DataFrame :
    ''' This functions converts calcite d18O (V-PDB standards) values to their drip water equivalent (V-SMOW standard), 
    using the mineralogy and temperature associated to each sample to convert.
    Input : 
        - caclite_df : DataFrame including columns d18O_meansurement (float64),T (float64), mineralogy (str)
    Outout :
        - converted_df : copy of calcite_df with an extra column d180p_VSMOW containing the drip water d18O V-SMOW values.
    '''
    # from literature (Comas-Bru 2019)
    conversion_cst = {'calcite'  :[16.1,24.6],
                      'aragonite':[18.34,31.954]
                      }
    converted_data = calcite_df.copy()
    converted_data['d18Op_VSMOW'] = pd.NA

    for mineralogy in ['calcite','aragonite']:
        mask = converted_data["mineralogy"]==mineralogy
        d18O = converted_data.loc[mask,'d18O_measurement']
        T = converted_data.loc[mask,'T']
        c0,c1 = conversion_cst[mineralogy]
        converted_data.loc[mask,'d18Op_VSMOW'] = 1.03092*d18O + 30.92 - ( c0*1000/T - c1 )
        print(f"for mineralogy {mineralogy}, the conversion failed for {converted_data.loc[mask,'d18Op_VSMOW'].isna().sum()} samples.")
    return converted_data

def retrieve_T_RegularGridInterp( data_df : DataFrame, temp_xda : DataArray, chrono : str , method : str = 'linear') -> DataFrame :
    ''' Retrieve the temperature of each sample of data_df by interpolating temp_xda values at data_df points using scipy RegularGridInterpolator
    Inputs :
        - data_df : DataFrame containing columns 'latitude', ' longitude', chrono. The chrono column should contain **positive** ages in yrs BP
        - temp_xda : DataArray containing a global temperature dataset with dimensions 'lat','lon','time'. 
                     Time is a **negative** age in yrs BP (i.e. -1000 stands for 1000 yrs BP).
        - chrono : str of the name of the chronology column in data_df. 
        - method : str of the name of the method to use to interpolate the temperature data points. Supported : "linear", "nearest", "slinear", "cubic", "quintic" and "pchip".
    Output : 
        - data_df : with an exra column 'T_interp' containing the temperature associated to each sample row.
    '''
    # set up the interpolator
    temp_lats  = temp_xda['lat'].values.copy()
    temp_lon   = temp_xda['lon'].values.copy()
    temp_times = temp_xda['time'].values.copy()
    temp_values = temp_xda.values

    interp_func = RegularGridInterpolator(
        points = (temp_times, temp_lats, temp_lon),
        values = temp_values,
        bounds_error = False,
        fill_value = np.nan,
        method = method
    )

    # set up sample points for interpolation
    sample_times = - data_df[chrono].values
    sample_lats = data_df['latitude'].values
    sample_lons = data_df['longitude'].values

    points = np.column_stack([sample_times, sample_lats, sample_lons])

    # interpolate
    data_df[f'T_{method}'] = interp_func(points)
    
    return data_df

# Deprecated : slower than RegularGridInterpolator, same results.
# def retrieve_T_nearest_neighbour(data_df : DataFrame, chrono : str, temp_xda : DataArray) -> DataFrame:
#     ''' This function aims to retrieve the surface temperature associated to each sample of the dataframe in input, 
#     using the global temperature dataset provided in input, by attributing the temperature value at the nearest point
#     (lon,lat,time) in this temperature dataset. 
#     Inputs :
#         - chrono : str representing a chronology method (e.g. lin_interp_age...) 
#         - data_df : DataFrame of data, including columns 'longitude', 'latitude' and chrono
#         - temp_xda : Xarray dataset providing temperature on a global grid with 'lat' and 'lon' coordinates. 
#     Output : 
#         - data_df : DataFrame of the data with an extra column T representing the surface temperature.
#     '''
#     print('-> retrieving temperatures using NN (takes a while...)')
#     # temp_xda.sel(lon=xr.DataArray(dataT['longitude'], dims="samples"),
#     #              lat=xr.DataArray(dataT['latitude'] , dims="samples"),
#     #              method='nearest',
#     #          ).values
#     # This one-liner produced OOM errors,
#     # and this method below was the slowest :
#     # dataT['T'] = [
#     #     temp_xda.sel(
#     #         lat=lat,
#     #         lon=lon,
#     #         time=-float(age),
#     #         method='nearest'
#     #     ).values.item()
#     #     for lat, lon, age in zip(dataT['latitude'], dataT['longitude'], dataT[chrono])
#     # ]
#     # so we prefer to apply this computation :
#     data_df['T'] = data_df.apply(lambda row : float(temp_xda.sel(lat = row['latitude'],
#                                                        lon = row['longitude'],
#                                                        time = -float(row[chrono]),
#                                                        method = 'nearest'
#                                                        )),axis=1)
#     print('temp NN retrieval done.')
#     return data_df

def retrieve_temperature_and_convert_speleothem_d18O(data_df : DataFrame, chrono : str, temp_xda : DataArray, method : str = 'linear')-> DataFrame :
    ''' 1) retrieves temperature of data_df samples based on the temp_xda datarray provided (interpolation with specified method + NN as backup) 
        2) convert speleothem PDB d18O into precipitation VSMOW d18O using Tremaine equation.
    Inputs :  TODO
    Outputs : TODO
    '''
    print("-> converting speleothem data")
    # 1. Temperature retrieval 
    data_to_convert = retrieve_T_RegularGridInterp(data_df = data_df,temp_xda = temp_xda, chrono = chrono, method = 'linear')
    #    Mask locs and times for which this method failed
    mask_nan = data_to_convert['T_linear'].isna()
    #    Apply the NN method for these points, if any
    if mask_nan.sum() :
        print(f'   {method} interpolation failed for {mask_nan.sum()} samples, trying to fill missing T with nearest neighbour method')
        data_to_convert['T_nearest'] = pd.NA
        data_to_convert.loc[mask_nan,'T_nearest'] = retrieve_T_RegularGridInterp(data_df=data_df[mask_nan].copy(),
                                                                                 temp_xda=temp_xda,
                                                                                 chrono=chrono,
                                                                                 method='nearest')['T_nearest']
    #    Gather T in a T column
    data_to_convert.loc[ mask_nan,'T'] = data_to_convert.loc[mask_nan,'T_nearest']
    data_to_convert.loc[~mask_nan,'T'] = data_to_convert.loc[~mask_nan,'T_linear']
    mask_nans_final = data_to_convert['T'].isna()
    print(f'   after {method} interpolation and nearest neighbour backup, still no temperature for', mask_nans_final.sum(),'samples.')
    print("   temperature retrieval finished, starting conversion")
    
    # 2. Conversion 
    converted_data = convert_calcite_to_drip_water(data_to_convert)
    print("conversion done.")

    return converted_data

################################################################################################# PLOTTING
def plot_global_map(data:DataFrame,
                    title:str,
                    quantity_col:str='d18O_measurement',
                    quantity:str='d18O',
                    unit:str='‰ VPDB',
                    proj:bool=True)-> Figure:
    ''' 3D or flat earth (natural earth proj)
    If proj=True : 2D 
    else : 3D
    '''
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=data["longitude"],
        lat=data["latitude"],
        text=data[quantity_col],
        mode="markers",
        marker=dict(
            symbol="triangle-up",
            size=10,
            color=data[quantity_col],
            colorscale="icefire",   # modern colormap
            # cmin=-15.5,
            # cmax=0,
            opacity=0.7,
            line=dict(color="white", width=1),
            colorbar=dict(
                title=f"{quantity}({unit})",
                ticks="outside",
                ticklen=6,
                thickness=15
            )
        )
    ))
    if proj :
        fig.update_layout(
            geo=dict(
                projection=dict(type="natural earth"),
                showland=True,
                landcolor="#f0f0f0",
                showocean=True,
                oceancolor="#dff4fd",
                showcountries=False,
                showcoastlines=True,
                showframe=False
            )
        )
    else :
        fig.update_layout(
            geo=dict(
                projection=dict(type="orthographic", rotation=dict(lat=12, lon=0)),
                showland=True,
                landcolor="#f0f0f0",
                showocean=True,
                oceancolor="#def4fd",
                showcountries=False,
                showcoastlines=False,
                showframe=False
            )
        )
        # fig.write_html("../output/mean_values_d18O_interactive_map_sisal.html", include_plotlyjs="cdn")
    fig.update_layout(
        title=dict(
                text=title,
                x=0.5,
                xanchor="center",
                font=dict(size=20, family="Arial, sans-serif")
            ),
            margin=dict(r=20, l=20, t=50, b=20),
            template="plotly_white"
    )
    return fig
