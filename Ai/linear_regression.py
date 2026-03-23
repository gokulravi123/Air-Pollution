import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Load Dataset
df = pd.read_csv('dataset.csv')

# 2. Preprocessing: Handle missing values (if any)
df = df.fillna(df.median())

# 3. Features & Target (Predicting CO level)
X = df.drop(columns=['CO'])
y = df['CO']

# 4. Split Data (80:20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predict and Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"--- Linear Regression Results ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
