print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Prediction for 6 hours:", model.predict([[6]])[0])