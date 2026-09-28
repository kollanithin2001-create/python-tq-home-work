import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame

print(df.head())
print(df.describe())
print(df["target"].value_counts())

df.drop(columns="target").hist(figsize=(8, 6))
plt.tight_layout()
plt.show()

for species in range(3):
    rows = df[df["target"] == species]
    plt.scatter(rows["petal length (cm)"], rows["petal width (cm)"],
                label=iris.target_names[species])

plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.legend()
plt.show()