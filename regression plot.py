import matplotlib.pyplot as plt

def plot_regression_results(model, X_test, y_test):
    # Predicting the values using the model
    y_pred = model.predict(X_test)
    
    # Plotting the predicted values against the actual values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
    plt.title('Multi-Linear Regression Results')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.show()

# Assuming 'model' contains the trained linear regression model
# Assuming 'X_test' and 'y_test' are the test set features and labels respectively
plot_regression_results(model, X_test, y_test)
