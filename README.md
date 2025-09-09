# Consumer Segments: K-Means Clustering
This project performs customer segmentation on 500k+ online retail transactions using K-Means clustering, leveraging data cleaning, feature scaling, and one-hot encoding to uncover purchase patterns and distinct customer segments



## Features
- Performs customer segmentation on 500k+ retail transactions
- Identifies purchasing behavior patterns with K-Means clustering
- **Source:** Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33. 
- **Features Used from Data:**
  - 'Quantity': number of items purchased
  - 'UnitPrice': price of each item
  - 'Country': customer country
- **Data Preprocessing**
  - Removed irrelevant columns (Description) and transactions missing a CustomerID
  - Removed outliers (top 1% in Quantity and UnitPrice)
  - Removed return transactions (negative numbers in Quantity)
  - Aggregated data per customer (average Quantity and UnitPrice)
  - Scaled numeric features using **StandardScaler**
  - One-hot encoded categorical features (Country)   



## Technologies
- **Python**
- **Pandas:** Data manipulation and analysis (loading, cleaning, aggregating)
- **NumPy:** Numerical operations (scaling)
- **Machine Learning:**
  - **K-Means Clustering:** Used scikit-learn for segmenting customers
  - **Silhouette Score:** Used to validate clusters
  - **Principal Component Analysis (PCA):** Used for visualization and reduced dimensionality
- **Visualizations**:
  - matplotlib for scatter plots
  - PCA to show clusters (reduced to 2 dimensions)



## Planned Improvements
- Leverage 'InvoiceDate' column to perform time-based analysis and check seasonal trends
- Produce customer segment profiles for actionable marketing insights


