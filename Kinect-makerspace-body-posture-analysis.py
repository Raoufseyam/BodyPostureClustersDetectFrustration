#!/usr/bin/env python
# coding: utf-8

# # Exploring Kinect Sensor readings of body postures as an indicator of student frustration levels in makerspaces
# 
# Mohamed Raouf Seyam
# 
# Harvard University
# 
# mohamedseyam@gse.harvard.edu

# **Overview**: 
# In this notebook, we investigate the association between frustraion and body postures as captured by 2 kinect sensors in a makerspace. The aim of this analysis is to examine the correlations between student reported levels of frustration, and clusters of body part (x,y) coordinates captured by the Kinect sensors. This notebook builds on previous analyses in Ramirez et al, 2019
# 
# **Participant privacy**
# All data used in this notebook was anonymized in a separate notebook
# 
# 

# ## Reading Data

# In[ ]:


# # for Google Colab use only
# from google.colab import drive
# drive.mount('content/')


# In[65]:


path = ''


# In[171]:


import re
import os
import csv
import math
import seaborn as sns
import pandas as pd
from scipy.stats.stats import pearsonr
from scipy import stats 
from scipy.stats import pearsonr
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[67]:


os.getcwd()


# In[68]:


os.listdir('../S435 - Kinect final project - raouf')


# Reading in student self-reported data and grouping by week

# In[69]:


data = pd.read_csv('../S435 - Kinect final project - raouf/all_survey_kinect_data.csv')
data = data.groupby('week', as_index=False).mean()


# Reading in  and combining weekly kinect sensor recordings

# In[123]:


files = os.listdir('./Kinect/')
print(files)


# In[124]:


master_df = 0
dfList = []
for filename in os.listdir('./Kinect'):
    if filename.endswith('.csv'):
        path = os.path.join('Kinect', filename)
        x = list(map(int, re.search('\d+', filename).group())) 
        df = pd.read_csv(path)
        df['week'] = ''.join(str(i) for i in x)
        print(len(df))
        if not type(master_df) == pd.core.frame.DataFrame:
            print('start')
            master_df = df
            dfList = [master_df]
        else:
            dfList.append(df)
    
        print('done with ' + filename)


# In[125]:


master_df = pd.concat(dfList)
master_df.shape


# ## Exploratory Data Analysis

# In[88]:


plt.plot(data.week, data.frustrated)
plt.xlabel('week')
plt.ylabel('frustrated')
plt.show


# In[87]:


plt.plot(data.week, data.actual_movement/1000)
plt.xlabel('week')
plt.ylabel('movement')
plt.show


# In[84]:


plt.plot(data.week, data.frustrated, 'bx--')
plt.xlabel('week')
plt.plot(data.week, data.actual_movement/1000, 'ro--')
plt.xlabel('week')
plt.ylabel('movement and frustration')
plt.show


# In[90]:


plt.scatter(data.frustrated, data.actual_movement)
plt.xlabel('frustration')
plt.ylabel('movement')
plt.show


# In[91]:


np.corrcoef(data.frustrated, data.actual_movement)


# In[103]:


plt.scatter(data.frustrated, data.head_to_elbwR)
plt.scatter(data.frustrated, data.head_to_elbwL)
plt.xlabel('frustration')
plt.ylabel('avg head to elbow distance(cm)')
plt.show


# In[108]:


plt.scatter(data.frustrated, data.handL_to_elbowR)
plt.scatter(data.frustrated, data.handR_to_elbowL)
plt.xlabel('frustration')
plt.ylabel('avg head to hand distance(cm)')
plt.show


# In[18]:


np.corrcoef(data.frustrated, data.actual_movement)


# In[110]:


np.corrcoef(data.frustrated, data.head_to_handL)


# In[111]:


np.corrcoef(data.frustrated, data.head_to_handR)


# In[126]:


dfcorr = data[['frustrated', 'week', 'hours_assignmt', 'actual_movement',
            'isTalking', 'head_to_handL','head_to_handR',
            'handL_to_elbowR', 'handR_to_elbowL']]
dfcorr.corr()


# In[130]:


pd.scatter_matrix(dfcorr, figsize=(6, 6))
plt.show()


# In[116]:


plt.matshow(dfcorr.corr())
plt.xticks(range(len(dfcorr.columns)), dfcorr.columns, rotation=90)
plt.yticks(range(len(dfcorr.columns)), dfcorr.columns, rotation=0)
plt.colorbar()
plt.show()


# In[139]:


survey = list(dfcorr.columns)[2:]
kinect = list(dfcorr.columns)[0:2]
print('Survey Data =====\n', survey, '\n')
print('Kinect Data =====\n', kinect)

correlations = dfcorr.corr()
fig, ax = plt.subplots(figsize=(5,4)) 
sns.heatmap(correlations.loc[survey][kinect], annot=True)


# # Cluster Analysis

# ## Kmeans

# In[38]:


# subset from master df
subset = pd.DataFrame(data=master_df[['HandRight_x', 'HandRight_y',
                                      'ElbowLeft_x', 'ElbowLeft_y',
                                      'HandLeft_x', 'HandLeft_y',
                                     'ElbowRight_x', 'ElbowRight_y']])
subset.shape


# In[40]:


# create a list of inertia values for k 1-10
from sklearn.cluster import KMeans

ks = list(range(1, 10))
inertias = []

for k in ks:
    
    # Create a KMeans instance with k clusters: model
    kmeans = KMeans(n_clusters=k, max_iter=1000)
    
    # Fit model to samples
    kmeans.fit(subset.values)
    
    # Append the inertia to the list of inertias
    inertias.append(kmeans.inertia_)


# In[41]:


# plot the inertia values using matplotlib
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# Scree plot supports a 3 cluster choice

# In[42]:


# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(subset)

# Determine the cluster labels of new_points: labels
master_df['cluster'] = model.predict(subset)


# In[43]:


# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.show()


# Creating a new dataframe that includes frustration/week and the number of instances spent in each of the clusters/week

# In[45]:


df_count = pd.DataFrame(columns = ['Week','Cluster0','Cluster1','Cluster2'])

df_count['Week'] = [2, 3, 4, 5, 6,  7, 9, 10, 11, 12, 13]

for i in range(2,8):
        df_count_temp = master_df.loc[master_df.week==str(i)]
        for j in range(3):
            df_count_temp1 = df_count_temp.loc[df_count_temp.cluster==j]
            df_count.iat[i-2,j+1]=df_count_temp1.shape[0]

for i in range(9,14):
        df_count_temp = master_df.loc[master_df.week==str(i)]
        for j in range(3):
            df_count_temp1 = df_count_temp.loc[df_count_temp.cluster==j]
            df_count.iat[i-3,j+1]=df_count_temp1.shape[0]


# Because week 12 is missing from the student self reported measures csv, it will be skipped in the new dataframe

# In[46]:


df_count['frustrated']=""
for i in range(0, 9):
    df_count['frustrated'][i] = dfcorr['frustrated'][i]

df_count['frustrated'][10] = dfcorr['frustrated'][9]


# In[47]:


df_count


# In[48]:


# removing week 12 due to missing frustration values
df_count.drop(df_count.index[9])


# Sanity check: added up the total culster values per week and matched to the total number of rows in each week's csv

# In[49]:


df_corr = pd.read_csv('../S435 - Kinect final project - raouf/df_corr.csv')


# In[50]:


df_corr


# In[51]:


np.corrcoef(df_corr.Week, df_corr.frustrated)


# In[52]:


plt.matshow(df_corr.corr())
plt.xticks(range(len(df_corr.columns)), df_corr.columns, rotation=90)
plt.yticks(range(len(df_corr.columns)), df_corr.columns, rotation=0)
plt.colorbar()
plt.show()


# In[59]:


survey = list(df_corr.columns)[4:]
kinect = list(df_corr.columns)[0:4]
print('Survey Data =====\n', survey, '\n')
print('Kinect Data =====\n', kinect)

correlations = df_corr.corr()
fig, ax = plt.subplots(figsize=(5,2)) 
sns.heatmap(correlations.loc[survey][kinect], annot=True)


# In[60]:


def calculate_pvalues(df):
    ''' computes the p-value for each correlation'''
    #df = df.dropna()._get_numeric_data()
    df = df._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            x,y = removeMissingData(df[r],df[c])
            results = stats.pearsonr(x,y)
            pvalues[r][c] = round(results[1], 4)
            #pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues

def removeNaN(a):
    return [x for x in a if not math.isnan(x)]

def removeMissingData(a, b):
    x = list(a)
    y = list(b)
    i = len(x) -1
    while(i != -1):  # get rid of missing values
        if x[i] == None or y[i] == None         or math.isnan(x[i]) or math.isnan(y[i]):
            del x[i]; del y[i]
        i -= 1
    return (x,y)


# In[62]:


p_values = calculate_pvalues(df_corr).astype(float)
fig, ax = plt.subplots(figsize=(5,2))
sns.heatmap(p_values.loc[survey][kinect], annot=True)


# ### DBScan 

# For computational speed (and because DBScan can be troublesome when n>100,000), only 1 every 100 rows will be seleced for a subsample of 7804

# In[180]:


subsample = [i for i in (0, len(subset)) if i % 100 == 0]


# In[176]:


subset = subsample.astype("float32", copy = False)

dbscan = DBSCAN(eps=0.5, min_samples=15).fit(subsample)

labels = dbscan.labels_

core_samples = np.zeros_like(labels, dtype = bool)
core_samples[dbscan.core_sample_indices_] = True

print("Number of points: %i" % label.size)


# In[ ]:


clf = NearestCentroid()
clf.fit(subset.values, label)

print(clf.centroids_.shape)


# ### NMF

# In[164]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

subset[subset.columns] = scaler.fit_transform(subset[subset.columns])

# Import NMF
from sklearn.decomposition import NMF

# Create an NMF instance: model
model = NMF(n_components=6)

# Fit the model to articles
model.fit(subset.values)

# Transform the articles: nmf_features
nmf_features = model.transform(subset.values)

# Print the NMF features
print(nmf_features)

