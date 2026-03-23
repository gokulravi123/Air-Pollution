import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load Dataset
df = pd.read_csv('dataset.csv')

# 2. Preprocessing: Handle missing values (if any)
df = df.fillna(df.median())

# 3. Features & Target (Classifying Air Quality)
X = df.drop(columns=['Air_Quality'])
y = df['Air_Quality']

# 4. Split Data (80:20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Predict and Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"--- Logistic Regression Results ---")
print(f"Accuracy Score: {accuracy:.4f}")
