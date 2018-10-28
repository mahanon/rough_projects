import pandas as pd
from pathlib import Path
import time

#######
#######
# Extract raw data from GBD files into naive database, save.

# Check code for config settings. For new data subset, define new config!
def gross_pd(config_no = 1, save_pickle = 1):
    print('Using config_no ' + str(config_no) + ', save_pickle = ' + str(save_pickle))
    if config_no == 1:
        print('\'year\', \'location_name\', \'cause_name\', \'val\' FOR all \'year\', all \'location_name\', \'age_name\' = \'All_Ages\', \'measure_name\' = \'Prevalence\', \'sex_name\' = \'Both\', \'metric_name\' = \'Rate\'')
    if config_no == 2:
        print('\'year\', \'location_name\', \'cause_name\', \'val\' FOR all \'year\', all \'location_name\', \'age_name\' = \'All_Ages\', \'measure_name\' = \'Incidence\', \'sex_name\' = \'Both\', \'metric_name\' = \'Rate\'')
    else:
        raise Exception('Put in config_no argument!')

    gdata_pd = pd.DataFrame()

    pathlist = Path('/Users/seanwoodward/Documents/datascience_stuff/gbd_data').glob('**/IHME*.csv')

    for path in pathlist: #itertools.islice(pathlist,0,2):
        data_pd = pd.read_csv(path)
    
        if config_no == 1:
            gdata_pd = pd.concat([gdata_pd,data_pd[['year','location_name','cause_name','val']][(data_pd['location_name'] != 'Global') & (data_pd['measure_name'] == 'Prevalence') & (data_pd['sex_name'] == 'Both') & (data_pd['age_name'] == 'All Ages') & (data_pd["metric_name"] == 'Rate')]],ignore_index=True)
        if config_no == 2:
            gdata_pd = pd.concat([gdata_pd,data_pd[['year','location_name','cause_name','val']][(data_pd['location_name'] != 'Global') & (data_pd['measure_name'] == 'Incidence') & (data_pd['sex_name'] == 'Both') & (data_pd['age_name'] == 'All Ages') & (data_pd["metric_name"] == 'Rate')]],ignore_index=True)
  
    # Save file named based on config.
    if save_pickle == 1:
        if config_no == 1:
            gdata_pd.to_pickle('gdata_config_01.pkl')
        if config_no == 2:
            gdata_pd.to_pickle('gdata_config_02.pkl')
    
    return gdata_pd
    
#######
#######
# Reformat data as shown in print statement below.
# !!! This is slow, conert to SQL for speedup if required again.

# Check code for config settings. For new data subset, define new config!
def feature_pd(config_no = 1, save_pickle = 1):
    print('Using config_no ' + str(config_no) + ', save_pickle = ' + str(save_pickle))
    if config_no == 1:
        print('\'year\', \'location_name\', \'cause_name\', \'val\' FOR all \'year\', all \'location_name\', \'age_name\' = \'All_Ages\', \'measure_name\' = \'Prevalence\', \'sex_name\' = \'Both\', \'metric_name\' = \'Rate\'')
        gdata_dir = 'gdata_config_01.pkl'
    elif config_no == 2:
        print('\'year\', \'location_name\', \'cause_name\', \'val\' FOR all \'year\', all \'location_name\', \'age_name\' = \'All_Ages\', \'measure_name\' = \'Incidence\', \'sex_name\' = \'Both\', \'metric_name\' = \'Rate\'')
        gdata_dir = 'gdata_config_02.pkl'
    
    else:
        raise Exception('Put in config_no argument!')
            
    gdata_pd = pd.read_pickle(gdata_dir).copy()
    
    gdata_causes_list = list(gdata_pd['cause_name'].unique())
    gdata_years_list = list(gdata_pd['year'].unique())
    gdata_years_list.sort()
    gdata_locations_list = list(gdata_pd['location_name'].unique())
    
    # initialize features frame
    fdata_pd = pd.DataFrame([[year] + [location] + [None]*len(gdata_causes_list) for year in gdata_years_list for location in gdata_locations_list], columns = ['year'] + ['location_name'] + gdata_causes_list) 
    
    t = time.time()
    print(len(gdata_pd.iloc[:,0]))
    for index,row in gdata_pd.iterrows():
        
        if index%10000 == 0:
            print(str(100*index/len(gdata_pd.iloc[:,0])) + '% of feature_pd job done in time ' + str(time.time() - t))
            t = time.time()   
        
        row_year = row['year']
        row_location = row['location_name']
        row_cause = row['cause_name']     
        row_val = float(row['val'])
        fdata_pd.loc[:,row_cause][(fdata_pd['year']==row_year) & (fdata_pd['location_name']==row_location)] = row_val
    
    # Save file named based on config.
    if save_pickle == 1:
        if config_no == 1:
            fdata_pd.to_pickle('fdata_config_01.pkl')
        elif config_no == 2:
            fdata_pd.to_pickle('fdata_config_02.pkl')
    
    return fdata_pd
 
#######
#######
# Perform extraction.

gdata_pd = gross_pd(2)
fdata_pd = feature_pd(2)
