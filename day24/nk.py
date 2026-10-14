from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset (4 features)
X = load_iris().data

# Reduce 4 features to 2
pca = PCA(n_components=2)
X_new = pca.fit_transform(X)

# Check explained variance
print("Variance:", pca.explained_variance_ratio_)

# Compare clusters before and after PCA
before = KMeans(n_clusters=3, random_state=42, n_init=10)
after = KMeans(n_clusters=3, random_state=42, n_init=10)

print("Before PCA:", before.fit_predict(X)[:10])
print("After PCA:", after.fit_predict(X_new)[:10])

# Visualize PCA clusters
plt.scatter(X_new[:, 0], X_new[:, 1], c=after.labels_)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA with KMeans")
plt.show()