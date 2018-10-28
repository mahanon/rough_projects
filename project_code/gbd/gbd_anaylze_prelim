import pandas as pd
import numpy as np
from fastcluster import linkage
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from plotly.plotly import iplot
from plotly.graph_objs import Scatter3d, Scatter, Data, Marker
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import scipy.spatial.distance as dst
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

# Data gbd_explore_01_make_pds.
fdata_pd = pd.read_pickle('fdata_config_01_first.pkl')

#######
#######
# Helper code.

# dist matrix sorting code modified from https://gmarti.gitlab.io/ml/2017/09/07/how-to-sort-distance-matrix.html
def seriation(Z,N,cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z
            
        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))
def compute_serial_matrix(dist_mat,method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)
        
        compute_serial_matrix transforms a distance matrix into 
        a sorted distance matrix according to the order implied 
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method,preserve_input=True)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]
    
    return seriated_dist, res_order, res_linkage

#######
#######
# Body functions.

# basic time series for single countries, single diseases
# Warning: I was experimenting with plotly
def plot_country_disease(country='Tanzania',disease='Malaria',data_pd=fdata_pd):
    country_slice_tanzania = data_pd.loc[(data_pd['location_name'].isin([country])),disease]
    trace = Scatter(x=list(range(1990,2017)),y=country_slice_tanzania)
    data = [trace]
    iplot(data, filename = 'basic-line')
    return None
    
# PCA and plotting of all countries in reduced, 3d disease space in different years
def plot_diseases_or_countries_3d(years=[2000],axis='disease',method='mds',outname='d_clusters_by_c_pattern_mds',data_pd=fdata_pd):
# axis is 'disease' or 'country'
# years is subset range 1990-2016
# method is 'pca' or 'mds'
    scaler = StandardScaler()
    
    if axis=='disease':
        year_slices = [scaler.fit_transform(data_pd.loc[(fdata_pd['year'].isin([year])),lambda s: s.columns[2:]].T) for year in years]
    elif axis=='country':
        year_slices = [scaler.fit_transform(data_pd.loc[(fdata_pd['year'].isin([year])),lambda s: s.columns[2:]]) for year in years]
    
    if method=='mds':
        red = MDS(n_components=3)
    elif method=='pca':
        red = PCA(n_components=3)
        
    # fit with full data
    all_year_slices = np.concatenate([year_slices[i] for i in range(len(year_slices))],axis=0)
    red.fit(all_year_slices)
    
    # transform individuals... could use above, not most efficient, can fix if time is issue.
    year_slices = [red.fit_transform(item) for item in year_slices]
    
    traces = [];
    for row in year_slices:
        traces.append([Scatter3d(x=year[:,0],y=year[:,1],z=year[:,2],mode='markers') for year in year_slices])
    
    data = Data(traces)
    iplot(data, filename = outname)
 
# Compute disease cosine distance matrices on disease trend gradients (Parsons-like Code) for all countries
# Un-cleaned, must be dealt with in subsequent steps.
def compute_all_country_cos_mats(outname='causes_cosine_matrices_prev_01.pkl', data_pd=fdata_pd):
    
    locations = data_pd['location_name'].unique()
    causes = data_pd.columns.values[2:]
    
    # index names: 'location_names','cause_1', 'cause_2'
    index = pd.MultiIndex.from_product([locations,causes,causes], names=['location_name','cause_1','cause_2'])
    data = []
    for location in locations:
        data += [[dst.cosine(np.gradient(data_pd.loc[(fdata_pd['location_name']==location),cause_1]),np.gradient(data_pd.loc[(fdata_pd['location_name']==location),cause_2]))] for cause_1 in causes for cause_2 in causes]
    causes_cosine_matrices_pd = pd.DataFrame(data,index=index)
    causes_cosine_matrices_pd.to_pickle('causes_cosine_matrices_prev_01.pkl')

    return causes_cosine_matrices_pd

# Average cosine disance matrices across countries, plot and return.
# It would appear doing this will-nilly erodes trends.
# Room for improvement.
def plot_cos_mat_across_countries(sim_method='average',read_file='causes_cosine_matrices_prev_01.pkl',data_pd=fdata_pd):
    # read_file from  compute_all_country_cos_mats
    # possible sim_method entries are 'average', 'ward', 'average'

    #locations = data_pd['location_name'].unique()
    causes = data_pd.columns.values[2:]
    
    # index names: 'location_names','cause_1', 'cause_2'
    causes_cosine_matrices_pd = pd.read_pickle('causes_cosine_matrices_prev_01.pkl')
    
    # group by cause_1, cause_2
    lgroup_causes_cosine_matrices_pd = causes_cosine_matrices_pd.groupby(['cause_1','cause_2'])
    
    # trimmean with 10% cut from tails
    trimmean_lgroup_cosmatrix_pd = lgroup_causes_cosine_matrices_pd.agg([lambda x: stats.trim_mean(x,0.1)])
    
    # create new matrix-style dataframe (reusing variable)
    trimmean_lgroup_cosmatrix_pd = pd.DataFrame(np.reshape(np.array(trimmean_lgroup_cosmatrix_pd),[322,322]),index=causes,columns=causes)
    
    #!!!  need to ensure all matrix diagonal values are zero, then replace NaN's with something...should consider solving NaN problem
    # Bad series are changed to max distance of 2.0,
    for cause in causes:
        trimmean_lgroup_cosmatrix_pd.at[cause,cause] = 0.0
    trimmean_lgroup_cosmatrix_array = np.array(trimmean_lgroup_cosmatrix_pd.fillna(2.0))
    
    # Reorder dissimilarity matrix by proximities.
    # Various methods available below... Tentatively using "average"
    # potential sim methods (for f argument) = ["ward","single","average","complete"]
    
    ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(trimmean_lgroup_cosmatrix_array,sim_method)
        
    plt.pcolormesh(ordered_dist_mat)
    plt.xlim([0,322])
    plt.ylim([0,322])
    plt.colorbar()
    
    plt.show()
    
    return trimmean_lgroup_cosmatrix_array

# Let's just look at Canada's cosine distance/covariance mat for now.
# Plot and return the cosine matrix in PD form.
# Later develop smarter way of combining.
def plot_cos_mat_one_country(location='Canada',sim_method='average', data_pd=fdata_pd):
    
    #locations = fdata_pd['location_name'].unique()
    causes = fdata_pd.columns.values[2:]
    
    index = pd.MultiIndex.from_product([causes,causes], names=['cause_1','cause_2'])
    
    #cosine dissimilarity
    data = [[dst.cosine(np.gradient(fdata_pd.loc[(fdata_pd['location_name']==location),cause_1]),np.gradient(fdata_pd.loc[(fdata_pd['location_name']==location),cause_2]))] for cause_1 in causes for cause_2 in causes]
    # For Testing, below: L2 dissimilarity 
    #data = [dst.pdist([np.gradient(fdata_pd.loc[(fdata_pd['location_name']==location),cause_1])/sum(fdata_pd.loc[(fdata_pd['location_name']==location),cause_1]),np.gradient(fdata_pd.loc[(fdata_pd['location_name']==location),cause_2])/sum(fdata_pd.loc[(fdata_pd['location_name']==location),cause_2])]) for cause_1 in causes for cause_2 in causes]    
    
    # Make dataframe
    cosmatrix_pd = pd.DataFrame(np.reshape(np.array(pd.DataFrame(data,index=index)),[322,322]),index=causes,columns=causes)
    
    #!!!  need to ensure all matrix diagonal values are zero, then replace NaN's with something...should consider solving NaN problem
    for cause in causes:
        cosmatrix_pd.at[cause,cause] = 0.0
    cosmatrix_pd_array = np.array(cosmatrix_pd.fillna(cosmatrix_pd.max().max()))
    
    ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(cosmatrix_pd_array,sim_method)
    
    plt.pcolormesh(ordered_dist_mat)
    plt.xlim([0,322])
    plt.ylim([0,322])
    plt.colorbar()
    
    plt.show()    
    
    # res_order is conversion back to original, unsorted indices.
    return res_order, cosmatrix_pd
    
# Below is used for plotting close disease clusters manually identified from
# the sorted cosine matrix
# Currently set to measles/TB sequence found for Canada.
def plot_trends_from_cos_ind(cosmatrix_indices=[7,8,9,10],res_order='Canada_TB',country='Canada',data_pd=fdata_pd):
    # Use 'Canada_TB' for res_order to bypass computations and use pre-found 
    # values. The default cosmatrix_indices and country also correspond to this set.
    
    if res_order=='Canada_TB':
        # Conversion previously done for Canada measles/TB
        pd_coords = [178+2, 75+2, 180+2, 181+2]
        labels = ['Multidrug-resistant tuberculosis without extensive drug resistance', 'Measles', \
                  'HIV-AIDS - Drug-susceptible Tuberculosis', 'HIV/AIDS - Multidrug-resistant Tuberculosis without extensive drug resistance']
    else:
        orig_coords = res_order(cosmatrix_indices)
        pd_coords = orig_coords + 2 # there is a difference between original np array and dataframe
        labels = data_pd.columns.values[pd_coords]

    fig, ax = plt.subplots()
    
    for i in range(len(pd_coords)):
        ax.plot(np.array(fdata_pd.loc[(fdata_pd['location_name'].isin([country])),lambda s: s.columns[[pd_coords[i]]]]),label=labels[i])
        
    ax.set_xticks(range(0,27,4))
    ax.set_xticklabels(range(1990,2017,4))
    ax.set_xlabel('Year')
    ax.set_ylabel('Cause Prevalence (per 100,000)')
    plt.legend(loc='upper right',fontsize='xx-small')
    plt.show()
    
#######
#######
# Demo.   
    
    
# Plot sorted cosine distance/covariance matrix for Canada.
# Closely related 7-10 correspond to TB/measles cluster potted, below.
plot_cos_mat_one_country()
    
# Make TB et al. plot for Canada
plot_trends_from_cos_ind()
