import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.DataFrame({
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Score': [20, 25, 35, 40, 65, 70, 85, 90]
})

wcss = []

for k in range(1, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(df)
    wcss.append(model.inertia_)

plt.plot(range(1, 7), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()