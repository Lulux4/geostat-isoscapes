import pandas as pd
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

def get_valid_sisal_samples_with_age(sisal_dict: dict, chrono : str ='lin_interp_age') -> dict :
    ''' Cleaning the table "sample_df" of the SISAL database to keep only valid samples, based on these criterions : 
    1) remove samples corresponding to superseded entities
    2) remove samples with mixed mineralogy, keep only calcite or aragonite speleothems
    3) remove samples with no "chrono" age where "chrono" refers to the chronology to use. 
    Inputs :
        - sisal_dict : dict of the sisal database as returned by function load_sisal()
        - chrono : str referring to the chronology method of interest. 
                   Samples with no age provided by this chrono methods will be excluded.
                   values : 'sisal-lin-interp','orig-chrono',... TODO
    Output :
        - vmin_vsample_vchrono_df : pandas dataframe containing the sample_df filtered for valid samples with valid age
        - filtered_chrono_df : pandas dataframe containing the chronoogy dataframe filtered for valid samples with valid age
    '''
    # Extract the dfs from sisal dict 
    entity_df = sisal_dict['entity']
    sample_df = sisal_dict['sample']
    if chrono == 'interp_age':
        chronology_df = sisal_dict['original_chronology']
    else :
        chronology_df = sisal_dict['sisal_chronology']

    # 1) Remove superseded entities and their samples
    ventity_df = entity_df.loc[entity_df['entity_status']!='superseded']
    vsample_df = sample_df[sample_df['entity_id'].isin(ventity_df['entity_id'].unique())]

    # 2) Exclude samples of mixed mineralogy entities :
    #   Find how many different mineralogies exist in the samples of each entity :
    unique_entities = vsample_df.groupby('entity_id')['mineralogy'].nunique() # dropna=False to count nan as a different value                                                                 
    #   Keep only those with a single mineralogy
    unique_min_sample_df = vsample_df[vsample_df['entity_id'].isin(unique_entities[unique_entities == 1].index)]
    vmin_vsample_df = unique_min_sample_df[unique_min_sample_df['mineralogy'].isin(['calcite', 'aragonite','secondary calcite'])]

    # 3) Keep only samples associated with a chronology age
    filtered_chrono_df = chronology_df[   ( chronology_df['sample_id'].isin(vmin_vsample_df['sample_id']) ) 
                                       & ~( pd.isna(chronology_df[chrono]) )
                                        ]
    vmin_vsamples_vchrono_array = filtered_chrono_df['sample_id'].unique()
    vmin_vsample_vchrono_df = vmin_vsample_df[vmin_vsample_df['sample_id'].isin(vmin_vsamples_vchrono_array)]

    # Final sample df is : 
    # merged_df = vmin_vsample_df[['entity_id','sample_id']].merge(filtered_chrono_df[['sample_id','lin_interp_age','lin_interp_age_uncert_neg','lin_interp_age_uncert_pos']], on='sample_id', how='left',)
    # merged_df = merged_df.merge(vmin_entity_df[['entity_id','site_id']], on='entity_id', how='left',)
    # data_df = merged_df.dropna(subset=['lin_interp_age'])#,'interp_age_uncert_neg','interp_age_uncert_pos'

    # print('We have',len(data_df['entity_id'].unique()),'speleothems with lin_interp age and pure mineralogy')
    # print(pure_entity_df.loc[~ pure_entity_df['entity_id'].isin(valid_df['entity_id']),['entity_id','site_id']])

    # update the sisal dict (maybe i should do it another way or return only the samples idx array?)
    return vmin_vsample_vchrono_df,filtered_chrono_df