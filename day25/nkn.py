import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Sample data
X = np.array([[10, 20], [12, 22], [14, 24],
              [16, 26], [100, 200]])

# 1. Detect outliers using IQR
Q1 = np.percentile(X, 25, axis=0)
Q3 = np.percentile(X, 75, axis=0)
IQR = Q3 - Q1

outliers = np.any((X < Q1 - 1.5 * IQR) |
                  (X > Q3 + 1.5 * IQR), axis=1)

print("Outliers:", X[outliers])

# 2. Detect far-away points
distance = np.linalg.norm(X - np.median(X, axis=0), axis=1)
print("Far points:", X[distance > 50])

# 3. Remove outliers
clean = X[~outliers]
print("Mean Before:", X.mean(axis=0))
print("Mean After:", clean.mean(axis=0))

# 4. KMeans and PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(clean)

model = KMeans(n_clusters=2, random_state=42, n_init=10)
print("Clusters:", model.fit_predict(X_pca))
print("PCA Variance:", pca.explained_variance_ratio_)