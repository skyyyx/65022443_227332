# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load Data
data = {
    'User_ID': [385, 681, 353, 895, 661, 846, 219, 588, 85, 465, 686, 408, 790, 116, 118, 54, 90, 372, 926, 94],
    'Gender': ['Male', 'Male', 'Male', 'Male', 'Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male', 'Male', 'Female', 'Female', 'Female', 'Female', 'Male', 'Male', 'Male', 'Female'],
    'Age': [35, 40, 49, 40, 25, 47, None, 42, 30, 41, 42, 47, 32, 27, 42, 33, None, 35, 46, 39],
    'AnnualSalary': [20000, 43500, None, 107500, 79000, 33500, 132500, 64000, 84500, None, 80000, 23000, 72500, 57000, 108000, 149000, 75000, 53000, 79000, 134000],
    'Purchased': ['No', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes']
}

df = pd.DataFrame(data)

# Preprocess Data
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Purchased'] = df['Purchased'].map({'No': 0, 'Yes': 1})
df = df.dropna()  # Drop rows with missing values

# Split Data
X = df[['Gender', 'Age', 'AnnualSalary']]
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Decision Tree Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate Model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

# Plot Decision Tree
plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.show()

# Plot Feature Importance
feature_importance = model.feature_importances_
features = X.columns
plt.bar(features, feature_importance)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()
