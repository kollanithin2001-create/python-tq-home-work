from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]  # Hours studied
y = [40, 50, 65, 70, 80]      # Actual marks

model = LinearRegression().fit(X, y)
predicted = model.predict(X)

print("Predictions:", predicted)