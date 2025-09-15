# -*- coding: utf-8 -*-
"""
Created on Nov 11 2023

This python script reads SISALv3 csv-data in a directory './SISALv3_csv' 
    relative to the path of this file (unpack all your downloaded csv-files 
    there!) and extracts stable isotope, Mg/Ca and growth rate data for all 
    entities, which cover a to be specified period of interest 
    (change lines 70 and 71 according to your needs). 

Only records with more than 'number_of_dating_points' U-Th dated depths
    (line 78) will be accounted for. 
    attention: it may happen that there are enough dated depth available in 
        the requested period, but proxies might not be provided this will 
        result in an empty output, but output nevertheless

The individual data will be plotted and the plots can be saved 
    (comment/uncomment line 225).

The mean and standard deviation of all proxies within your specified period 
    will be determined and saved in a csv file.

There will also be a raw plot for illustrative purposes available.

Feel free to change the code as you see fit.
    
@author: Jens Fohlmeister
"""

import numpy as np
import pandas as pd
import os
import pygmt
import matplotlib.pyplot as plt
plt.close("all")    # close all open figures  

###########################################################################
# read all sisalv3.csv files
###########################################################################
os.chdir("data/sisalv3_csv")

entity = pd.read_csv('entity.csv')
d13C = pd.read_csv("d13C.csv")
d18O = pd.read_csv("d18O.csv")
MgCa = pd.read_csv("Mg_Ca.csv")
dating = pd.read_csv("dating.csv")
# some columns of the database have a number on the first position of their name which would produce errors in python
dating.rename(columns = {'238U_content':'c238U_content','238U_uncertainty':'c238U_uncertainty',
    '232Th_content':'c232Th_content','c232Th_uncertainty':'c232Th_uncertainty',
    '230Th_content':'c230Th_content','c230Th_uncertainty':'c230Th_uncertainty',
    '230Th_232Th_ratio':'a230Th_232Th_ratio','230Th_232Th_ratio_uncertainty':'a230Th_232Th_ratio_uncertainty',
    '230Th_238U_activity':'a230Th_238U_activity','230Th_238U_activity_uncertainty':'a230Th_238U_activity_uncertainty',
    '234U_238U_activity':'a234U_238U_activity','234U_238U_activity_uncertainty':'a234U_238U_activity_uncertainty'},
    inplace = True)

entity_link_reference = pd.read_csv("entity_link_reference.csv")
original_chronology = pd.read_csv("original_chronology.csv")
reference = pd.read_csv("reference.csv")
sample = pd.read_csv("sample.csv")
sisal_chronology = pd.read_csv("sisal_chronology.csv")
site = pd.read_csv("site.csv")
os.chdir('../')
###########################################################################


###########################################################################
# extract required data from speleothems covering the period of interest
#   + provides all entities, which include non-14C ages and non-events  
#     during the time period
###########################################################################
low = 0         # defines minimum age [a]
high = 500000    # defines maximum age [a] # 500'000 = U/Th datation limit, 50'000 = C14 limit
# Corrected age = corr_age = adjusted for detrital contamination
# Corrected calibrated age = c14 date_type = adjusted for dead carbon
i0 = dating.loc[(dating['corr_age'] >= low) & (dating['corr_age'] <= high) &
                 (dating['date_type'].str.find('Event')!=0)] # (dating['date_type']!='C14') &  ## Jens excluded C14 dates but for now i do not see why i should exclude them
print(f'number of samples with C14 dates is {len(i0.loc[dating['date_type']!='C14'])/len(i0)}')

i1 = i0['entity_id'].to_numpy()
i3 = np.unique(i1) 

## remove all entities with less than 'number_of_dating_points' dated depths
number_of_dating_points = 3
j=0
for i in np.arange(0,len(i3)):
    i_dummy = i0.entity_id[i0.entity_id==i3[i]].count()
    if i_dummy < number_of_dating_points:
        j=j+1
        i0 = i0[i0.entity_id!=i3[i]]
print(f'{j} entities were removed (not enough dated depths)')

i1 = i0['entity_id'].to_numpy()
i2 = np.unique(i1)  # provides all entities, which include >= 'number_of_dating_points' 
                    # dated depths during the required time period
###########################################################################


### define parameters (all of those will be saved in a final file)
site1_id = np.zeros(len(i2))
site_name1 = ['0']*len(i2)
rock_age1 = ['0']*len(i2)
material1 = ['0']*len(i2)
entity_name1 = ['0']*len(i2)
lon = np.zeros(len(i2))
lat = np.zeros(len(i2))
entity1_id = np.zeros(len(i2))
mean_C = np.zeros(len(i2))
mean_O = np.zeros(len(i2))
mean_GR = np.zeros(len(i2))
mean_MgCa = np.zeros(len(i2))
std_C = np.zeros(len(i2))
std_O = np.zeros(len(i2))
std_GR = np.zeros(len(i2))
std_MgCa = np.zeros(len(i2))

print('number of entities : ',len(i2))
for n in np.arange(0,len(i2)):
    plt.close("all")    # close all open figures  

    dummy = dating.loc[(dating['entity_id'] == i2[n])]
    #######################################################################

    ### already some metadata for individual speleothems
    site1_id[n] = entity.site_id[(entity['entity_id'] == i2[n])].to_numpy()
    entity1_id[n] = entity.entity_id[(entity['entity_id'] == i2[n])].to_numpy()
    entity_name1[n] = entity.entity_name[(entity['entity_id'] == i2[n])].to_list()
    site_name1[n] = site.site_name[(site['site_id'] == site1_id[n])].to_list()
    lon[n] = site.longitude[(site['site_id'] == site1_id[n]).to_numpy()]
    lat[n] = site.latitude[(site['site_id'] == site1_id[n]).to_numpy()]
    if dummy.material_dated.dropna().eq('calcite').all():
        material1[n] = 'calcite'
    elif dummy.material_dated.dropna().eq('aragonite').all():
        material1[n] = 'aragonite'
    else:
        material1[n] = 'mixed'
    print("Number:", n, entity_name1[n])
    
    ### extract isotope data (d18O and d13C) and elements #####################
    idx1 = sample.sample_id[(sample['entity_id']==i2[n])].to_numpy()
    age = original_chronology.interp_age[original_chronology['sample_id'].isin(idx1)].to_numpy()
    idx2 = original_chronology.sample_id[original_chronology['sample_id'].isin(idx1)].to_numpy()
    d18O_1 = d18O.d18O_measurement[d18O['sample_id'].isin(idx2)].to_numpy()
    idx3 = d18O.sample_id[d18O['sample_id'].isin(idx2)].to_numpy()
    age18 = original_chronology.interp_age[original_chronology['sample_id'].isin(idx3)].to_numpy()
    if len(idx3) < len(idx2):
        idx2 = idx3
        age = original_chronology.interp_age[original_chronology['sample_id'].isin(idx2)].to_numpy()
            
    d13C_1 = d13C.d13C_measurement[d13C['sample_id'].isin(idx2)].to_numpy()
    idx4 = d13C.sample_id[d13C['sample_id'].isin(idx2)].to_numpy()
    age13 = original_chronology.interp_age[original_chronology['sample_id'].isin(idx4)].to_numpy()
    MgCa_1 = MgCa.Mg_Ca_measurement[MgCa['sample_id'].isin(idx2)].to_numpy()
    idx5 = MgCa.sample_id[MgCa['sample_id'].isin(idx2)].to_numpy()
    ageMgCa = original_chronology.interp_age[original_chronology['sample_id'].isin(idx5)].to_numpy()

    ### also growth rate (gr) could be important ##############################
    if len(idx2) != 0:
        isotopeDepth = sample.depth_sample[sample['sample_id'].isin(idx2)].to_numpy()
        gr = np.zeros(len(isotopeDepth))
        for i in np.arange(0,len(gr)-1):
            if entity.depth_ref[(entity['entity_id'] == i2[n])].to_numpy() == 'from top':
                gr[i] = (isotopeDepth[i] - isotopeDepth[i+1]) / (age[i] - age[i+1])
            else:
                gr[i] = -(isotopeDepth[i] - isotopeDepth[i+1]) / (age[i] - age[i+1])
                
        gr[-1] = gr[-2]
        if len(np.argwhere(np.isinf(gr))>0): 
            if (np.argwhere(np.isinf(gr))[-1]==len(gr)-1): # if the last value is 'inf'
                gr[np.argwhere(np.isinf(gr))[-1]] = gr[np.argwhere(np.isinf(gr))[-1]-2]
            else:
                gr[np.argwhere(np.isinf(gr))]=gr[np.argwhere(np.isinf(gr))+1] # replace 'inf' values by neighboring values for gr
            while len(np.argwhere(np.isinf(gr))>0): # second iteration for cases where there is very fast growth and initially two successive 'inf' values
                gr[np.argwhere(np.isinf(gr))]=gr[np.argwhere(np.isinf(gr))+1] # replace 'inf' values by neighboring values for gr

        for i in np.arange(1,len(gr)-1):
            if gr[i]>1: 
                gr[i]=(gr[i-1]+gr[i+1])/2
                
        ### determine the according averages  #################################
        if len(d18O_1)>0:
            mean_GR[n] = np.mean(gr[np.argwhere((age18>=low) & (age18<=high))])
            mean_O[n] = np.mean(d18O_1[np.argwhere((age18>=low) & (age18<=high))])
            std_GR[n] = np.std(gr[np.argwhere((age18>=low) & (age18<=high))])
            std_O[n] = np.std(d18O_1[np.argwhere((age18>=low) & (age18<=high))])
        if len(d13C_1)>0:
            mean_C[n] = np.mean(d13C_1[np.argwhere((age13>=low) & (age13<=high))])
            std_C[n] = np.std(d13C_1[np.argwhere((age13>=low) & (age13<=high))])
        if len(MgCa_1)>0:
            mean_MgCa[n] = np.mean(MgCa_1[np.argwhere((ageMgCa>=low) & (ageMgCa<=high))])
            std_MgCa[n] = np.std(MgCa_1[np.argwhere((ageMgCa>=low) & (ageMgCa<=high))])
        #######################################################################
            
        ### define plot
    #     fig,ax = plt.subplots(3,1, num='Isotopes+Elements; Entity '+str(int(entity1_id[n])),figsize = (10, 7.5),clear = True)

    #     ### top plot (MgCa) ###################################################
    #     ax[0].plot(ageMgCa,MgCa_1,'-g')
    #     ax[0].set_xlim(low,high)
    #     ax[0].set_ylabel('Mg/Ca ratio [ ]', color='g', fontsize = 15)
    #     ax[0].tick_params(axis='y', colors='green', labelsize = 12)
    #     ax[0].tick_params(axis='x', labelsize = 12)
    #     ax[0].set_title(str("Entity_id: " + str(i2[n]) +" (" +
    #             entity.entity_name[(entity['entity_id'] == i2[n])].to_numpy() + ", " +
    #             str(site.site_name[(site['site_id'] == site1_id[n])])[5:-31] +
    #             ")")[2:-2],fontsize = 20, x=0.5, y=1.3)
    #     ax[0].xaxis.tick_top()
    #     #######################################################################

    #     ### plot isotopes in center subplot
    #     ax[1].plot(age,d18O_1,'-r')
    #     if len(d13C_1)==len(age):
    #         ax3 = ax[1].twinx()
    #         ax3.plot(age13,d13C_1,'-b')
    #         ax3.tick_params(axis='y', colors='blue', labelsize = 12)
    #         ax3.set_ylabel(r"$\delta^{13}$C " + u"[\u2030 VPDB]", color='b', fontsize = 15) #use u"[\u2030]" for permil sign

    #     ax[1].set_ylabel(r"$\delta^{18}$O " + u"[\u2030 VPDB]", color='r', fontsize = 15)
    #     ax[1].set_xlim(low,high)
    #     ax[1].tick_params(axis='y', colors='red', labelsize = 12)
    #     ax[1].set_xticks([])
    #     ###################################################################
        
        
    #     ### bottom plot (growth rate after isotope depth) #################
    #     #ax[2].stairs(age,gr,'k')
    #     ax[2].stairs(gr[0:-1],age, color='black', baseline=None)
    #     ax[2].set_ylabel('growth rate \n [mm/a]', fontsize = 15)
    #     ax[2].set_xlabel('age [a BP]', fontsize = 15)
    #     ax[2].set_xlim(low, high)
    #     ax[2].tick_params(axis='both', labelsize = 12)
    #     ax[2].xaxis.set_label_coords(.5, -.2)
    #     ###################################################################

    # fig.subplots_adjust(wspace=0, hspace=0, top=0.85)
    # fig.savefig('tests/'+ str(i2[n]) +'_'+ entity.entity_name[(entity['entity_id'] == i2[n])].to_list()[0] +'.png',dpi=300)


### save file and produce an overview plot
output = pd.DataFrame({'site_id':site1_id, 'site_name':site_name1,
                        'longitude':lon, 'latitude':lat,
                        'entity_id':entity1_id, 'eintity_name':entity_name1,
                        'material':material1, 
                        'mean_GR [mm/a]':mean_GR, 'std_GR [mm/a]':std_GR,
                        'mean_d13C':mean_C, 'std_d13C':std_C,
                        'mean_d18O':mean_O, 'std_d18O':std_O,
                        'mean_MgCa':mean_MgCa, 'std_MgCa':std_MgCa})
output.replace(0, np.nan, inplace=True)
# with pd.ExcelWriter('isotope_data_'+str(np.round(low/1000).astype(int))+'-'+str(np.round(high/1000).astype(int))+'ka.xlsx', engine='xlsxwriter') as writer:  
#     output.to_excel(writer, sheet_name= 'Period '+str(np.round(low/1000).astype(int))+'-'+str(np.round(high/1000).astype(int))+' ka')
output.to_csv('tests/output_test.csv')


### plotting mean_d18O on a world map via pyGMT
### you might have to install pygmt by one of the following options:
#    conda create --name pygmt --channel conda-forge pygmt
#    mamba create --name pygmt --channel conda-forge pygmt 
#    pip install pygmt (via spyder)


out1 = output.dropna(subset='mean_d18O') # remove all 'nan' in mean d18O for plotting purposes
fig = pygmt.Figure()
# Use region "d" to specify global region (-180/180/-90/90)
fig.coast(region="d", projection="N12c", land="grey", water="snow2", frame="afg")
pygmt.makecpt(series=[-15.5,0], cmap="polar")
fig.plot(
        x=out1.longitude,
        y=out1.latitude,
        style="c0.3c",#"c",   # '0.3c' is for inverted triangles of 0.3cm size
        #size=2*np.ones(len(output.latitude)),
        fill=out1.mean_d18O,
        cmap = True,
        transparency=50,  # set transparency level for all symbols
        pen="black"
    )
fig.colorbar(frame="af+ld18O (permil VPDB)")
# fig.show()
fig.savefig('tests/output_map.png',dpi=300)

import plotly.graph_objects as go
fig = go.Figure()

fig.add_trace(go.Scattergeo(
    lon=out1["longitude"],
    lat=out1["latitude"],
    text=out1["mean_d18O"],
    mode="markers",
    marker=dict(
        symbol="triangle-up",
        size=10,
        color=out1["mean_d18O"],
        colorscale="icefire",   # modern colormap
        cmin=-15.5,
        cmax=0,
        opacity=0.7,
        line=dict(color="white", width=1),
        colorbar=dict(
            title="δ18O (‰ VPDB)",
            ticks="outside",
            ticklen=6,
            thickness=15
        )
    )
))

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
    ),
    title=dict(
        text="Global δ18O Distribution",
        x=0.5,
        xanchor="center",
        font=dict(size=20, family="Arial, sans-serif")
    ),
    margin=dict(r=20, l=20, t=50, b=20),
    template="plotly_white"
)
fig.write_html("tests/map_plotly_interactive.html", include_plotlyjs="cdn")
fig.show()

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

fig.write_image("tests/map_plotly.png", scale=3)
