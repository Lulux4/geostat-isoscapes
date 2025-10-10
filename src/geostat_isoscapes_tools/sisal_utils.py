import pandas as pd
from pandas import DataFrame
import os 

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

def clean_sisal_data(sisal_dict: dict, chrono : str ='lin_interp_age') -> dict :
    ''' Cleaning the table "sample_df" of the SISAL database to keep only valid samples, based on these criterions :
    1.  remove samples corresponding to superseded entities
    2.  remove samples with mixed mineralogy, keep only calcite or aragonite speleothems
    3.  remove samples with no "chrono" age where "chrono" refers to the chronology to use. 
    4.  remove samples for which we do not have a d18O measurement
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
    sample_df1 = sample_df[sample_df['entity_id'].isin(entity_df1['entity_id'].unique())]
    #############################
    # 2) Exclude samples of mixed mineralogy entities :
    #   Find how many different mineralogies exist in the samples of each entity :
    unique_entities = sample_df1.groupby('entity_id')['mineralogy'].nunique() # dropna=False to count nan as a different value                                                                 
    #   Keep only those with a single mineralogy
    unique_min_sample_df = sample_df1[sample_df1['entity_id'].isin(unique_entities[unique_entities == 1].index)]
    sample_df2 = unique_min_sample_df[unique_min_sample_df['mineralogy'].isin(['calcite', 'aragonite','secondary calcite'])]
    #############################
    # 3) Keep only samples associated with a chronology age
    chrono_df1 = chronology_df[( chronology_df['sample_id'].isin(sample_df2['sample_id']) ) 
                               & ~( pd.isna(chronology_df[chrono]) )
                               ]
    sample_df3 = sample_df2[sample_df2['sample_id'].isin(chrono_df1['sample_id'])]
    #############################
    # 4) Keep only samples existing in d18O_df and having a non nan measurement
    d18O_df1 = d18O_df.dropna(subset='d18O_measurement')
    sample_df4 = sample_df3[(sample_df3['sample_id'].isin(d18O_df1['sample_id']))]


    final_entity_df = entity_df[entity_df['entity_id'].isin(sample_df4['entity_id'])]
    final_chrono_df = chronology_df[chronology_df['sample_id'].isin(sample_df4['sample_id'])]
    final_d18O_df = d18O_df[d18O_df['sample_id'].isin(sample_df4['sample_id'])]
    final_site_df = site_df[site_df['site_id'].isin(final_entity_df['site_id'])]
    
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
        - merged_df : DataFrame containing the columns specified in list arguments and the columns of indices 'sample_id','entity_id' and 'site_id'.
    '''

    entities1 = entity_df[ col_entity+['site_id','entity_id']].merge(site_df[ col_site+['site_id'] ],on='site_id',how='left')
    sample1 = sample_df[ col_sample+['entity_id','sample_id'] ].merge(entities1, on='entity_id',how='left')
    sample2 = sample1.merge(chrono_df[ col_chrono+['sample_id'] ], on='sample_id',how='left')
    merged_df = sample2.merge(d18O_df[ col_d18O+['sample_id'] ], on='sample_id', how='left')

    return merged_df

def get_basic_cleaned_merged_sisal_data(chrono : str ='interp_age')-> DataFrame:
    # Load SISAL data
    print('loading database')
    sisal_dict = load_sisal()

    # Clean the data
    print('cleaning samples')
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
    print('merging dataframes')
    col_site = ['site_name','longitude','latitude']
    col_entity = []
    col_chrono = [chrono]
    col_sample = ['mineralogy']
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
    print('returning merged dataframe')
    return merged_data