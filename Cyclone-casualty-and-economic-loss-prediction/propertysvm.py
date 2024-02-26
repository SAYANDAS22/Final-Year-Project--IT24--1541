import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load the CSV file
data = pd.read_csv('property_data.csv')

# Separate the features (X) and target variable (y)
X = data.drop('property_loss', axis=1)
y = data['property_loss']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Support Vector Regression (SVR) model
svr_model = SVR(kernel='linear')  

# Train the model
svr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svr_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


print("\nPredicted Property Loss Values:")
print(y_pred)


# Plot the actual vs predicted values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Property Loss')
plt.ylabel('Predicted Property Loss')
plt.title('SVR: Actual vs Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Perfect prediction line
plt.show()
