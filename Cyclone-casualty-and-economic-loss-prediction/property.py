import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the CSV file
data = pd.read_csv('property_data.csv')

# Separate the features (X) and target variable (y)
X = data.drop('property_loss', axis=1)
y = data['property_loss']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Make predictions on new data
new_data = pd.read_csv('new_property_data.csv')
new_predictions = rf_model.predict(new_data)
print("Predictions for new data:")
print(new_predictions)