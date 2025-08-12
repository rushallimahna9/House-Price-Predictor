import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load Kaggle dataset
df = pd.read_csv("train.csv")
# will keep only the relevant column
df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]

# Features and target
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model to file
pickle.dump(model, open('model.pkl', 'wb'))

print("✅ Model trained and saved as model.pkl")



