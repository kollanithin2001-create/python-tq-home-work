import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("iris")

# 1. Count plot, box plot, and pair plot
sns.countplot(data=df, x="species")
plt.show()

sns.boxplot(data=df, x="species", y="petal_length")
plt.show()

sns.pairplot(df, hue="species")
plt.show()

# 2. Numerical distributions and mean versus median
print(df.select_dtypes("number").agg(["mean", "median"]))
sns.histplot(data=df, x="petal_length", kde=True)
plt.show()

# 3. Correlation heatmap
sns.heatmap(df.select_dtypes("number").corr(), annot=True)
plt.show()

# 4. Category averages
sns.barplot(data=df, x="species", y="petal_length")
plt.show()