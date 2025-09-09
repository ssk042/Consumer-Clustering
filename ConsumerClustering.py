# Online Retail Customer Segmentation (Aggregated)
# ---------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ---------------------------
# 1. Load Data
# ---------------------------
df = pd.read_excel('Online Retail.xlsx')

# ---------------------------
# 2. Data Cleaning
# ---------------------------

# Drop 'Description' (not needed)
df = df.drop(columns=['Description'])

# Drop rows without CustomerID
df = df.dropna(subset=['CustomerID'])

# Keep only positive quantities (remove returns)
df = df[df['Quantity'] > 0]

# Remove outliers in Quantity and UnitPrice (top 1%)
df = df[df['Quantity'] < df['Quantity'].quantile(0.99)]
df = df[df['UnitPrice'] < df['UnitPrice'].quantile(0.99)]

# Filter countries with enough transactions
country_counts = df['Country'].value_counts()
valid_countries = country_counts[country_counts > 100].index
df = df[df['Country'].isin(valid_countries)]

# ---------------------------
# 3. Feature Engineering
# ---------------------------

# One-hot encode Country column
df = pd.get_dummies(df, columns=['Country'])
one_hot_cols = [col for col in df.columns if col.startswith('Country_')]

# Aggregate per customer
df_customer = df.groupby('CustomerID').agg({
    'Quantity': 'mean',       # average quantity per purchase
    'UnitPrice': 'mean',      # average price per purchase
    **{col: 'max' for col in one_hot_cols}  # country presence
}).reset_index()

# ---------------------------
# 4. Scaling
# ---------------------------
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_customer[['Quantity', 'UnitPrice']]),
                         columns=['Quantity', 'UnitPrice'])

# Combine scaled numerics with one-hot encoded country columns
df_final = pd.concat([df_scaled, df_customer[one_hot_cols].reset_index(drop=True)], axis=1)

# ---------------------------
# 5. K-Means Clustering
# ---------------------------
k = 4  # can adjust after elbow/silhouette
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(df_final)

# Add cluster labels
df_final['Cluster'] = kmeans.labels_

# Cluster counts
print("Number of customers in each cluster:")
print(df_final['Cluster'].value_counts())

# Silhouette score
score = silhouette_score(df_final.drop('Cluster', axis=1), df_final['Cluster'])
print(f'Silhouette Score for K={k}: {score:.4f}')

# ---------------------------
# 6. Visualize Clusters
# ---------------------------
# PCA to reduce to 2D
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_final.drop('Cluster', axis=1))

plt.figure(figsize=(8,6))
plt.scatter(df_pca[:,0], df_pca[:,1], c=df_final['Cluster'], cmap='viridis', s=20, alpha=0.7)
plt.title('Customer Segments (K-Means Clustering)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()