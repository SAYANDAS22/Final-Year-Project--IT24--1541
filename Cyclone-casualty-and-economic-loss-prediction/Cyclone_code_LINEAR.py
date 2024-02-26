# Import necessary libraries
import pandas as pd
from datacleaner import autoclean
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split , cross_val_score

# Function to preprocess dataset
def preprocess_dataset(dataset_name, df, exclude_columns=None):
    # Remove excluded columns
    if exclude_columns is not None:
        # Identify excluded columns
        excluded_cols = df.columns[df.columns.isin(exclude_columns)]
        # Remove excluded columns
        df_to_clean = df.drop(columns=exclude_columns)
    else:
        # Use all columns for cleaning
        df_to_clean = df

    # Clean the data automatically (with default strategies)
    cleaned_df = autoclean(df_to_clean)

    # Save the cleaned data
    cleaned_df.to_csv(f"{dataset_name}1.csv", index=False)

    # Scale numerical features to have a mean of 0 and a standard deviation of 1
    scaler = StandardScaler()
    cleaned_df_scaled = pd.DataFrame(scaler.fit_transform(cleaned_df), columns=cleaned_df.columns)

    # Save the cleaned data
    cleaned_df_scaled.to_csv(f"{dataset_name}1.csv", index=False)

    # Normalize numerical features to the range [0, 1]
    scaler = MinMaxScaler()
    cleaned_df_normalized = pd.DataFrame(scaler.fit_transform(cleaned_df_scaled), columns=cleaned_df_scaled.columns)


    # Concatenate excluded columns with the normalized dataframe
    cleaned_df_normalized = pd.concat([df[excluded_cols], cleaned_df_normalized], axis=1)

    # Save the cleaned data
    cleaned_df_normalized.to_csv(f"{dataset_name}1.csv", index=False)

    return cleaned_df_normalized

# Load and process the first dataset
df1 = pd.read_csv("Population.csv")
exclude_columns = ['STATE']
processed_df1 = preprocess_dataset("Population", df1, exclude_columns=exclude_columns)
print("Processed Population dataset:\n", processed_df1.head())

# Load and process the second dataset
df2 = pd.read_csv("Gdp.csv")
exclude_columns = ['STATE']
processed_df2 = preprocess_dataset("Gdp", df2, exclude_columns=exclude_columns)
processed_df2['STATE']=processed_df2['STATE'].str.upper()
print("\nProcessed Gdp dataset:\n", processed_df2.head())

# Load and process the third dataset
df3 = pd.read_csv("Historical.csv")
exclude_columns = ['WIND SPEED', 'PLACE','YEAR']
processed_df3 = preprocess_dataset("Historical", df3, exclude_columns=exclude_columns)
processed_df3['PLACE']=processed_df3['PLACE'].str.upper()
print("\nProcessed Historical dataset:\n", processed_df3.head())



# Merge Population and Historical datasets using STATE and PLACE columns
merged_df = pd.merge(processed_df1, processed_df3, left_on='STATE', right_on='PLACE', how='inner')

# Drop the PLACE column from the merged dataframe
merged_df = merged_df.drop(columns=['PLACE'])


# Print the merged dataframe
print("\nMerged Population and Historical dataset:\n", merged_df.head())
#Save the merge dataset 
merged_df.to_csv('demo_data12.csv',index=False)


df = pd.get_dummies(merged_df, columns=['STATE'])
# Assuming 'target_column' is your target variable (label) and you want to predict it
X = df.drop(['monetory loss', 'CASUALITIES'], axis=1)
y = df['CASUALITIES']

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)


# Import libraries for linear regression and plotting
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Create linear regression model 
lin_reg = LinearRegression()

# Fit the model on the training data
lin_reg.fit(X_train, y_train)

# Make predictions on test data
y_pred = lin_reg.predict(X_test)

# Plot results
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Linear Regression Model_Cyclone Prediction')

# Calculate metrics
from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print('R-squared:', r2) 
print('MAE:', mae)
plt.show()

new_data = pd.DataFrame({
    'POPULATION': [100000],
    'AREA': [500],
    'YEAR': [2022],
    'monetory loss': [10000],
    'WIND SPEED': [1],
    'STATE': ['AL']
})
# Preprocess the new data
new_data = pd.get_dummies(new_data, columns=['STATE','WIND SPEED']).drop(['monetory loss'], axis=1)

# Rename the features of the new data to match those of the training data
new_data = new_data.rename(columns={'AREA': 'Area (In sq. km)', 'POPULATION': 'Females Population', 'STATE_AL': 'STATE', 'WIND SPEED_1': 'WIND SPEED'})

# Make a prediction using the trained model
prediction = lin_reg.predict(new_data)

# Print the prediction
print("Prediction: ", prediction)

"""
new_data = pd.DataFrame({'POPULATION': [100000], 'AREA': [500], 'YEAR': [2022], 'monetory loss': [10000], 'WIND SPEED': [1], 'STATE': ['AL']})
# Preprocess the new data
new_data = pd.get_dummies(new_data, columns=['STATE','WIND SPEED'])\
              .drop(['monetory loss'], axis=1)

# Rename the features of the new data to match those of the training data
new_data = new_data.rename(columns={'AREA': 'Area (In sq. km)', 'POPULATION': 'Females Population', 'STATE_AL': 'STATE', 'WIND SPEED_1': 'WIND SPEED'})

# Make a prediction using the trained model
prediction = lr_model.predict(new_data)

# Print the prediction
print("Prediction: ", prediction)

"""