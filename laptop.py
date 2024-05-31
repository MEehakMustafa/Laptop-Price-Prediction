import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
file_path = 'laptopPrice.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(df.head())

# Display basic statistics of the dataframe
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Convert the 'rating' column to numeric values
df['rating'] = df['rating'].str.extract('(\d)').astype(float)

# Handling missing values for rating, Number of Ratings, and Number of Reviews
imputer = SimpleImputer(strategy='mean')
df['rating'] = imputer.fit_transform(df[['rating']])
df['Number of Ratings'] = imputer.fit_transform(df[['Number of Ratings']])
df['Number of Reviews'] = imputer.fit_transform(df[['Number of Reviews']])

# Encoding categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split the data into features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

# Example of predicting the price of a new laptop
new_laptop = {
    'brand': 'HP',
    'processor_brand': 'Intel',
    'processor_name': 'Core i5',
    'processor_gnrtn': '10th',
    'ram_gb': 8,
    'ram_type': 'DDR4',
    'ssd': 512,
    'hdd': 0,
    'os': 'Windows',
    'os_bit': 64,
    'graphic_card_gb': 2,
    'weight': 1.75,
    'warranty': 1,
    'Touchscreen': 'No',
    'msoffice': 'Yes',
    'rating': 4.5,
    'Number of Ratings': 150,
    'Number of Reviews': 20
}

# Encode the new laptop data
new_laptop_encoded = {key: label_encoders[key].transform([value])[0] if key in label_encoders else value for key, value in new_laptop.items()}

# Convert to DataFrame
new_laptop_df = pd.DataFrame([new_laptop_encoded])

# Predict the price
predicted_price = model.predict(new_laptop_df)
print(f'Predicted Price: {predicted_price[0]}')
