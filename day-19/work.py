from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# 1. Load data
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Try different max_depth values
for depth in [1, 2, 3, 4]:
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(X_train, y_train)

    print("Depth:", depth)
    print("Accuracy:", model.score(X_test, y_test))

# 3. Generate predictions
predictions = model.predict(X_test)
print("Predictions:", predictions)

# Feature importance
plt.bar(data.feature_names, model.feature_importances_)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.show()