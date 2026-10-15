model = KMeans(n_clusters=2, random_state=42, n_init=10)

df['Cluster'] = model.fit_predict(df)

print(df)
print(df.groupby('Cluster').mean())