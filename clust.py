from sklearn.preprocessing import StandardScaler

X = df[['Hours', 'Score']]

before = KMeans(n_clusters=2, random_state=42, n_init=10)
df['Before'] = before.fit_predict(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

after = KMeans(n_clusters=2, random_state=42, n_init=10)
df['After'] = after.fit_predict(X_scaled)

print(df[['Hours', 'Score', 'Before', 'After']])