import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# loading data file
df = pd.read_excel('Online Retail.xlsx')

# Data Clean-Up
# handling the missing data
# dropping description column because has empty values and isn't necessary for clustering
df = df.drop(columns=['Description'])

# dropping blank customer ID rows because want to focus on loyal customers
df = df.dropna(subset=['CustomerID'])

# Remove outliers of Quantity and Unit Price not in 99th percentile
# Removing top 1%, so clusters are not skewed
df = df[df['Quantity'] < df['Quantity'].quantile(0.99)]
df = df[df['UnitPrice'] < df['UnitPrice'].quantile(0.99)]

# only want positive values (removing returns)
df = df[df['Quantity'] > 0]

# Want to focus on countries with substantial amount of transactions (100+)
count_of_countries = df['Country'].value_counts() # counting country transactions
approved_countries = count_of_countries[count_of_countries > 100].index
df = df[df['Country'].isin(approved_countries)]


# One-Hot Encoding
# converting country column to multiple binary cols, row has 1 under correct country 
df = pd.get_dummies(df, columns=['Country']) 
one_hot_cols = [col for col in df.columns if 'Country_' in col]

# averages of customers, average qty customers bought and average price per item 
df_customer = df.groupby('CustomerID').agg({
    'Quantity': 'mean',          # mean quantity per transaction (using aggregate function)
    'UnitPrice': 'mean',         # mean price per item across all transactions -> customer behavior
    **{col: 'max' for col in one_hot_cols} 
    # dict of dummy country columns -> if customer purchased from given country, max = 1
}).reset_index() # CustomerID as a regular column again, not used for indexing 


# Scaling for K-Means Clustering
# need to protect cluster distances
# standard scaling best for distance basis of k-means: mean = 0, std dev = 1
scaler = StandardScaler()

# calculating mean and std dev of each column + applying scaling
df_scaled = scaler.fit_transform(df_customer[['Quantity', 'UnitPrice']])

# converting numpy array to pandas df
df_scaled = pd.DataFrame(df_scaled, columns=['Quantity', 'UnitPrice'])

# concatenating countries horizontally to scaled df
# essentially combining scaled data with one-hot encoded
newdf = pd.concat([df_scaled, df_customer[one_hot_cols].reset_index(drop=True)], axis=1) 


# K-Means Clustering
k = 4 # checked with silhouette score
kmeans = KMeans(n_clusters=k, random_state=0) # clustering model
kmeans.fit(newdf)    # trains model in new df

# labeling clusters to see clusters customers belong to
newdf['Cluster'] = kmeans.labels_ 
print("Number of customers in each cluster:")
print(newdf['Cluster'].value_counts())

# calculating silhouette score -> silhouette_score(features, labels)
score = silhouette_score(newdf.drop(('Cluster'), axis=1), newdf['Cluster'])
print(f'Silhouette Score for K={k}: {score:.4f}')

# Clustering Visual
# using principal component analysis to reduce to 2D feature space
# represent linear combo of orginal features 
pca = PCA(n_components=2) # data varies in diff directions
pcadf = pca.fit_transform(newdf.drop('Cluster', axis=1)) # each pc = new combo of features given

# adding color, labels/adjusting optics
plt.figure(figsize=(8,6))
plt.scatter(pcadf[:,0], pcadf[:,1], c=newdf['Cluster'], cmap='viridis', s=20, alpha=0.7)
plt.title('Consumer Segments: K-Means Clustering')
plt.xlabel('PCA Component 1') # direction where variance is highest
plt.ylabel('PCA Component 2') # direction where variance is 2nd highest
plt.colorbar(label='Cluster')
plt.show()



