#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
df=pd.read_excel("Custom_CNN_Features1.xlsx")
df


# In[2]:


dataframe=df.drop(columns=['Filename'])
print(dataframe)


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt

#Load the dataset
df = pd.read_excel("Custom_CNN_Features1.xlsx")

#Loading 2 features having numeric values
feature_a = df['f0']
feature_b = df['f6']

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(feature_a, feature_b, color='blue', alpha=0.5)

# Add labels and title
plt.xlabel('f0')
plt.ylabel('f6')
plt.title('Scatter Plot of f0 vs f6')

# Show plot
plt.grid(True)
plt.show()


# In[6]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel("Custom_CNN_Features1.xlsx")

# Assume 'embed_1' is the independent variable
independent_variable = df['f6']
dependent_variable = df['f0']

# Reshape the data as sklearn's LinearRegression model expects 2D array
independent_variable = independent_variable.values.reshape(-1, 1)
dependent_variable = dependent_variable.values.reshape(-1, 1)

# Create a linear regression model
model = LinearRegression()
model.fit(independent_variable, dependent_variable)

# Predict the values
predicted_values = model.predict(independent_variable)

# Calculate mean squared error
mse = mean_squared_error(dependent_variable, predicted_values)
print(f'Mean Squared Error: {mse}')

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(independent_variable, dependent_variable, color='blue', alpha=0.5)

# Plot the regression line
plt.plot(independent_variable, predicted_values, color='red')

# Add labels and title
plt.xlabel('f0')
plt.ylabel('f6')
plt.title('Linear Regression Model: f0 vs f6')

# Show plot
plt.grid(True)
plt.show()


# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#Load the dataset
df = pd.read_excel("Custom_CNN_Features1.xlsx")

# Assuming X_train, y_train, X_test, y_test are your training and test sets

# Initialize the Logistic Regression model
logistic_model = LogisticRegression()

binary_dataframe = df[df['Label'].isin([0, 1])]
X = binary_dataframe[['f6', 'f0']]
y = binary_dataframe['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Logistic Regression model on the training data
logistic_model.fit(X_train, y_train)

# Use the trained model to make predictions on the test set
predictions = logistic_model.predict(X_test)

# Calculate accuracy by comparing predicted labels to actual labels in the test set
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of Logistic Regression on the test set: {accuracy * 100:.2f}%")


# In[8]:


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_excel("Custom_CNN_Features1.xlsx")

# Assuming your target variable is 'target_variable'
target_variable = df['Label']

# Extracting features
X = df[['f0', 'f6']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, target_variable, test_size=0.2, random_state=42)

# Decision Tree Regressor
reg_tree = DecisionTreeRegressor(random_state=42)
reg_tree.fit(X_train, y_train)
y_pred_tree = reg_tree.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
print(f"Decision Tree Mean Squared Error: {mse_tree}")

# k-NN Regressor
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train_scaled, y_train)
y_pred_knn = knn_regressor.predict(X_test_scaled)
mse_knn = mean_squared_error(y_test, y_pred_knn)
print(f"k-NN Regressor Mean Squared Error: {mse_knn}")


# In[ ]:


#dataset no:-2


# In[3]:


import pandas as pd
data1=pd.read_csv('palm_document_Gabor.csv')
print(data1.head())


# In[7]:


import pandas as pd

# Load the Excel file into a DataFrame
df = pd.read_csv('palm_document_Gabor.csv')

# Define a function to change values in the "Label" column
def change_label(row):
    if row['Label'] == 'bad':
        return 2
    if row['Label']== 'good':
        return 0
    if row['Label']== 'medium':
        return 1

# Apply the function to the "Label" column
df['Label'] = df.apply(change_label, axis=1)

# Save the modified DataFrame back to an Excel file
df.to_excel('modified_dataset.xlsx', index=False)
print(df.head(15))


# In[10]:


import pandas as pd
import matplotlib.pyplot as plt

#Load the dataset
df = pd.read_csv("palm_document_Gabor.csv")

#Loading 2 features having numeric values
feature_a = df['Theta0_Lambda1_LocalEnergy']
feature_b = df['Theta0_Lambda1_MeanAmplitude']

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(feature_a, feature_b, color='blue', alpha=0.5)

# Add labels and title
plt.xlabel('Theta0_Lambda1_LocalEnergy')
plt.ylabel('Theta0_Lambda1_MeanAmplitude')
plt.title('Scatter Plot of Theta0_Lambda1_LocalEnergy vs Theta0_Lambda1_MeanAmplitude')

# Show plot
plt.grid(True)
plt.show()


# In[11]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("palm_document_Gabor.csv")

# Assume 'embed_1' is the independent variable
independent_variable = df['Theta0_Lambda1_MeanAmplitude']
dependent_variable = df['Theta0_Lambda1_LocalEnergy']

# Reshape the data as sklearn's LinearRegression model expects 2D array
independent_variable = independent_variable.values.reshape(-1, 1)
dependent_variable = dependent_variable.values.reshape(-1, 1)

# Create a linear regression model
model = LinearRegression()
model.fit(independent_variable, dependent_variable)

# Predict the values
predicted_values = model.predict(independent_variable)

# Calculate mean squared error
mse = mean_squared_error(dependent_variable, predicted_values)
print(f'Mean Squared Error: {mse}')

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(independent_variable, dependent_variable, color='blue', alpha=0.5)

# Plot the regression line
plt.plot(independent_variable, predicted_values, color='red')

# Add labels and title
plt.xlabel('Theta0_Lambda1_LocalEnergy')
plt.ylabel('Theta0_Lambda1_MeanAmplitude')
plt.title('Linear Regression Model: Theta0_Lambda1_LocalEnergy vs Theta0_Lambda1_MeanAmplitude')

# Show plot
plt.grid(True)
plt.show()


# In[19]:


import pandas as pd

# Load the Excel file into a DataFrame
df = pd.read_csv('palm_document_Gabor.csv')

# Define a function to change values in the "Label" column
def change_label(row):
    if row['Label'] == 'bad':
        return 2
    if row['Label']== 'good':
        return 0
    if row['Label']== 'medium':
        return 1

# Apply the function to the "Label" column
df['Label'] = df.apply(change_label, axis=1)

# Save the modified DataFrame back to an Excel file
df.to_excel('modified_dataset.xlsx', index=False)
print(df.head(10))


# In[20]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#Load the dataset
# Assuming X_train, y_train, X_test, y_test are your training and test sets

# Initialize the Logistic Regression model
logistic_model = LogisticRegression()

binary_dataframe = df[df['Label'].isin([0, 1])]
X = binary_dataframe[['Theta0_Lambda1_MeanAmplitude', 'Theta0_Lambda1_LocalEnergy']]
y = binary_dataframe['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Logistic Regression model on the training data
logistic_model.fit(X_train, y_train)

# Use the trained model to make predictions on the test set
predictions = logistic_model.predict(X_test)

# Calculate accuracy by comparing predicted labels to actual labels in the test set
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of Logistic Regression on the test set: {accuracy * 100:.2f}%")


# In[21]:


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the dataset
# Assuming your target variable is 'target_variable'
target_variable = df['Label']

# Extracting features
X = df[['Theta0_Lambda1_LocalEnergy','Theta0_Lambda1_MeanAmplitude']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, target_variable, test_size=0.2, random_state=42)

# Decision Tree Regressor
reg_tree = DecisionTreeRegressor(random_state=42)
reg_tree.fit(X_train, y_train)
y_pred_tree = reg_tree.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
print(f"Decision Tree Mean Squared Error: {mse_tree}")

# k-NN Regressor
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train_scaled, y_train)
y_pred_knn = knn_regressor.predict(X_test_scaled)
mse_knn = mean_squared_error(y_test, y_pred_knn)
print(f"k-NN Regressor Mean Squared Error: {mse_knn}")


# In[ ]:




