import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

customers = pd.DataFrame({
    'Income': [20, 25, 30, 60, 65, 70, 90, 95],
    'Spending': [15, 20, 25, 50, 55, 60, 85, 90]
})

model = KMeans(n_clusters=3, random_state=42, n_init=10)
customers['Cluster'] = model.fit_predict(customers)

print(customers)

plt.scatter(customers['Income'],
            customers['Spending'],
            c=customers['Cluster'])

plt.xlabel('Income')
plt.ylabel('Spending')
plt.title('Customer Clusters')
plt.show()