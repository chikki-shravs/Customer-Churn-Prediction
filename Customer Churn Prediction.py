#!/usr/bin/env python
# coding: utf-8

# ## Predicting Churn Customers

# ### Business Question: 
# How can we reduce customer churn in the bank's credit card services by identifying customers at risk of leaving and providing them with targeted services to encourage them to stay?

# ### Data Description:
# 
# A manager at the bank is disturbed with more and more customers leaving their credit card services. They would really appreciate it, if one could predict for them who is going to get churned so they can proactively go to the customer to provide them better services and turn customers decisions in the opposite direction. 
Variable_Name	Description
Client_Number:	Unique identifier for the customer holding the account
Attrition_Flag:	This variable indicates whether a customer is an Existing Customer or has Churned.
Customer_Age:	Customer’s age.
Gender:	Customer's gender.
Dependent_count:	The number of dependents the customer has.
Education_Level:	Educational Qualification of the account holder (high school, college, graduate, etc.)
Marital_Status:	The customer's marital status.
Income_Category: 	The customer's income category. (Married, Single, Divorced, Unknown)
Card_Category: 	The category of the customer's credit card. (Blue, Silver, Gold, Platinum)
Months_on_book: 	How long the customer has been a cardholder.
Total_Relationship_Count: 	The total number of products held by the customer.
Months_Inactive_12_mon: 	Number of months with no transactions in the last 12 months.
Contacts_Count_12_mon: 	Number of contacts with the customer in the last 12 months.
Credit_Limit: 	Credit limit on the card.
Total_Revolving_Bal: 	Total revolving balance on the card.
Avg_Open_To_Buy:	 Average open-to-buy credit line.
Total_Amt_Chng_Q4_Q1: 	Change in transaction amount Q4 2019 to Q1 2020.
Total_Trans_Amt: 	Total transaction amount in the last 12 months.
Total_Trans_Ct: 	Total transaction count in the last 12 months.
Total_Ct_Chng_Q4_Q1: 	Change in transaction count Q4 2019 to Q1 2020.
Avg_Utilization_Ratio: 	Average card utilization ratio.
# # Load libraries

# In[1]:


# Import the libraries 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


# Data Preprocessing Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Machine Learing (classification models) Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, roc_curve, roc_auc_score 
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold


import warnings
warnings.filterwarnings("ignore")


# ## Read the dataset

# In[2]:


# Load the dataset
data = pd.read_csv("C:/Users/sravani/Downloads/BankChurners.csv/BankChurners.csv")


# ## Data Exploration

# In[3]:


# No of rows and columns
data.shape


# In[4]:


# Calculate summary statistics for numeric columns
data.describe()


# In[5]:


# List of columns to remove
columns_to_remove = ['CLIENTNUM','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2']

# Remove the specified columns
data = data.drop(columns=columns_to_remove)


# CLIENTNUM is a unique identifier it does not help much in prediction of the models.

# In[6]:


# Splitting columns into Categorical and Numerical Feature Lists 
categorical_features = [
    'Attrition_Flag', 'Gender', 'Education_Level', 'Marital_Status',
    'Income_Category', 'Card_Category'
]

numerical_features = [
    'Customer_Age', 'Dependent_count', 'Months_on_book', 
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
]


# The target variable is "Attrition_Flag," which has two classes: "Existing Customer" and "Attrited Customer." A binary classification problem involves predicting one of two possible outcomes for a given input.

# In[7]:


# Perform exploratory data analysis (EDA)
# Visualize the distribution of features and their relationships with customer churn

# Plot the distribution of the target variable
sns.countplot(x='Attrition_Flag', data=data)
plt.title("Customer Churn Distribution")
plt.show()


# In[8]:


data['Attrition_Flag'].value_counts()


# Countplot shows the distribution of the target variable "Attrition_Flag," which represents customer churn ("Existing Customer" or "Attrited Customer"). It counts the number of instances in each category and visualizes it as a bar chart.

# In[9]:


# Explore feature distributions and relationships
# Example: Boxplot of customer age by churn status
#Explore the relationships between the "Attrition_Flag" (churn status) and the "Customer_Age" feature using a boxplot.
sns.boxplot(x='Attrition_Flag', y='Customer_Age', data=data)
plt.title("Customer Age by Churn Status")
plt.show()


# This boxplot displays the distribution of customer ages for both existing and attrited customers.
# The boxplot allows to compare the distribution of ages for both customer groups. 

# In[10]:


# Define a custom color palette
#Countplot to visualize the distribution of income categories in the dataset
custom_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
plt.figure(figsize=(9, 4))
sns.countplot(data=data, x='Income_Category', palette=custom_palette, edgecolor='black')
plt.title("Income Category Distribution", fontsize=14)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# In[11]:


# The distribution of genders,
gender_counts = data['Gender'].value_counts()

# Define custom colors
custom_colors = ['#FF9999', '#66B2FF']

# Create labels with count
labels = [f"{gender} ({count})" for gender, count in zip(gender_counts.index, gender_counts)]

plt.pie(gender_counts, labels=labels, autopct='%1.1f%%', colors=custom_colors)
plt.title('Gender Distribution', fontsize=14)
plt.show()


# The counts for each gender category.Female count is 5358 and male count is 4768.

# In[12]:


# Define a custom color palette
custom_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

plt.figure(figsize=(9, 4))
sns.countplot(data=data, x='Education_Level', palette=custom_palette, edgecolor='black')
plt.title("Education Level Distribution", fontsize=14)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# In[13]:


# Define a custom color palette
custom_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
#The distribution of marital statuses in the dataset
plt.figure(figsize=(9, 4))
sns.countplot(data=data, x='Marital_Status', palette=custom_palette, edgecolor='black')
plt.title("Marital Status Distribution", fontsize=14)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# In[14]:


# Define a custom color palette
custom_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
#The distribution of card category in the dataset
plt.figure(figsize=(9, 4))
sns.countplot(data=data, x='Card_Category', palette=custom_palette, edgecolor='black')
plt.title("Card Category Distribution", fontsize=14)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# In[15]:


categorical_columns = ['Gender', 'Marital_Status', 'Income_Category', 'Card_Category','Education_Level']



# In[16]:


# Select the categorical columns you want to visualize
categorical_columns = ['Gender', 'Marital_Status', 'Income_Category', 'Card_Category','Education_Level']

# Create subplots to display multiple count plots
fig, axes = plt.subplots(nrows=len(categorical_columns), ncols=1, figsize=(8, 12))

# Iterate through the selected categorical columns
for i, column in enumerate(categorical_columns):
    sns.countplot(x=column, hue='Attrition_Flag', data=data, ax=axes[i], palette='viridis')
    axes[i].set_title(f'Countplot of {column} by Attrition_Flag')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Count')

# Adjust layout
plt.tight_layout()
plt.show()


# In[17]:


# Counts for different categories within each of the selected categorical columns, 
# categorized by the "Attrition_Flag," which represents whether the customer has churned or not. 
# Create cross-tabulations to calculate counts
for column in categorical_columns:
    cross_tab = pd.crosstab(data[column], data['Attrition_Flag'])
    print(f'Counts for {column} by Attrition_Flag:')
    print(cross_tab)
    print('\n')


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt

# Select numerical features
numerical_features = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

# Create a new DataFrame with only numerical features and the 'Attrition_Flag' column
data_numerical = data[numerical_features + ['Attrition_Flag']]

# Create side-by-side box plots
plt.figure(figsize=(14, 8))
sns.set(style="whitegrid")

for feature in numerical_features:
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Attrition_Flag', y=feature, data=data_numerical)
    plt.title(f'Distribution of {feature} by Attrition_Flag')
    plt.xticks(rotation=45)
    plt.show()

# Display value counts for each numerical feature by Attrition_Flag
for feature in numerical_features:
    counts = data_numerical.groupby(['Attrition_Flag', feature]).size()
    print(f'Value Counts for {feature}:')
    print(counts)
    print('\n' + '-'*40 + '\n')


# # Data Pre-Processing

# In[19]:


#Find if null values exist in the dataset
null_values = data.isna().sum()
# Display the count of null values for each column
print(null_values)


# Dataset does not contain any missing values.

# In[20]:


print(data.columns)


# In[21]:


# Working with Ordinal Features with pandas `map` method.

attrition_flag_dictionary = {'Existing Customer' : 0, 'Attrited Customer' : 1}

edu_level_dictionary = {'Unknown': 0, 'Uneducated': 1, 'High School': 2, 'College': 3
                 , 'Post-Graduate': 4, 'Graduate': 5, 'Doctorate': 6} 

income_cat_dictionary = {'Unknown': 0, 'Less than $40K': 1, '$40K - $60K': 2, '$60K - $80K': 3
                  , '$80K - $120K': 4 ,'$120K +': 5}

card_cat_dictionary = {'Blue': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}

data['Attrition_Flag'] = data['Attrition_Flag'].map(attrition_flag_dictionary)

data['Education_Level'] = data['Education_Level'].map(edu_level_dictionary)

data['Income_Category'] = data['Income_Category'].map(income_cat_dictionary)

data['Card_Category'] = data['Card_Category'].map(card_cat_dictionary)

data.head()


# In[22]:


# Working with Nominal Features with pandas `get_dummies` function.
data = pd.get_dummies(data, columns=['Gender', 'Marital_Status'])

encoded = list(data.columns)
print("Features after one-hot encoding.".format(len(encoded)))


# ### Handling Outliers

# In[23]:


# Calculating the first quartile (Q1), third quartile (Q3), and interquartile range (IQR) for numerical features
Q1 = data[numerical_features].quantile(0.25)
Q3 = data[numerical_features].quantile(0.75)
IQR = Q3 - Q1

# Setting a threshold for identifying outliers (e.g., 2.5 times the IQR)
threshold = 2.5

# Calculating the lower bound and upper bound for outliers based on the threshold
lower_bound = Q1 - threshold * IQR
upper_bound = Q3 + threshold * IQR


# In[24]:


# Detecting Outliers in each column 
for column in numerical_features:
    column_outliers = data[(data[column] < lower_bound[column]) | (data[column] > upper_bound[column])]
    print(f"Feature '{column}' has == {len(column_outliers)} outliers.")


# In[25]:


# Create a boolean mask to identify rows containing outliers
outliers_mask = ((data[numerical_features] < lower_bound) | (data[numerical_features] > upper_bound)).any(axis=1)

print(f"Dataset Before Minimizing Outliers {data.shape}\n")

# Remove rows with outliers from the DataFrame
data = data[~outliers_mask]

# Display the DataFrame without outliers
print(f"Dataset After Minimizing Outliers {data.shape}")


# ### Split the data into train and test sets 

# In[26]:


# Split the data into training and testing sets
X = data.drop('Attrition_Flag', axis=1)
y = data['Attrition_Flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[27]:


X.shape, y.shape


# In[28]:


y_train.value_counts()


# ### Handle Imbalanced Data

# In[29]:


from imblearn.over_sampling import RandomOverSampler

# Initialize the RandomOverSampler
oversampler = RandomOverSampler(random_state=42)

# Fit and transform the training data
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Show the results after oversampling
print("Training set after oversampling has {} samples".format(len(y_train_resampled)))
print(y_train_resampled.value_counts())


# In[30]:


# Creating a StandardScaler instance
scaler = StandardScaler()

# Fitting the StandardScaler on the training data
scaler.fit(X_train[numerical_features])

# Transforming (standardize) the continuous features in the training and testing data
X_train_cont_scaled = scaler.transform(X_train[numerical_features])
X_test_cont_scaled = scaler.transform(X_test[numerical_features])

# Replacing the scaled continuous features in the original data
X_train[numerical_features] = X_train_cont_scaled
X_test[numerical_features] = X_test_cont_scaled

X_train


# ### Feature Scaling

# ### Models training and Evaluation

# In[31]:


# Below is the classification models we will be using
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": xgb.XGBClassifier()
}


# In[32]:


# Creating lists for classifier names, test_accuracy_scores and their F1 scores.
classifier_names = []
test_accuracy_scores = []
f1_scores = []

# Looping through classifiers, fitting models, and calculating train and test accuracy
for name, model in classifiers.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    classifier_names.append(name)
    
    # Calculating and storing F1 score
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)
    
    # Calculating and storing test accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    test_accuracy_scores.append(test_accuracy)

    # Printing model details
    print(f'Model: {name}')
    print(f'Training Accuracy: {accuracy_score(y_train, model.predict(X_train))}')
    print(f'Testing Accuracy: {accuracy_score(y_test, y_pred)}')
    print('------------------------------------------------------------------')
    print(f'Testing Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
    print('------------------------------------------------------------------')
    print(f'Testing Classification report: \n{classification_report(y_test, y_pred)}')
    print('------------------------------------------------------------------')


# ### Test Accuracy Scores by Classifiers

# In[33]:


# Sample data
data = pd.DataFrame({'Classifier': classifier_names, 'Test Accuracy': test_accuracy_scores})

# Sort the DataFrame by 'Test Accuracy' in descending order
data = data.sort_values(by='Test Accuracy', ascending=False)

# Define custom colors for each classifier
custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create a vertical bar plot using Seaborn
plt.figure(figsize=(9, 5))
ax = sns.barplot(data=data, x='Classifier', y='Test Accuracy', palette=custom_colors)
plt.title('Test Accuracy Scores by Classifiers')
plt.xlabel('Classifier')
plt.ylabel('Test Accuracy')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

plt.show()


# ### F1-Score by Classifiers

# In[34]:


# Sample data
data = pd.DataFrame({'Classifier': classifier_names, 'F1-Score': f1_scores})

# Sort the DataFrame by 'F1-Score' in descending order
data = data.sort_values(by='F1-Score', ascending=False)

# Create a vertical bar chart using Seaborn
plt.figure(figsize=(9, 5))
ax = sns.barplot(data=data, x='Classifier', y='F1-Score', palette='viridis')
plt.title('F1-Score by Classifiers')
plt.xlabel('Classifier')
plt.ylabel('F1-Score')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

plt.show()


# In[35]:


# Fitting the XGBClassifier on training data
xgboost = xgb.XGBClassifier()
xgboost.fit(X_train, y_train)


# In[36]:


#  Predict churn probability using the trained XGBoost model
y_pred_prob = xgboost.predict_proba(X_test)[:, 1]  # Probabilities for the positive class (churn)

# Set a threshold for churn (default is 0.5)
churn_threshold = 0.5

# Identify customers with a churn probability higher than the threshold
churned_customers = X_test[y_pred_prob > churn_threshold]

# Add churn probabilities to the filtered customers for better insight
churned_customers['Churn_Probability'] = y_pred_prob[y_pred_prob > churn_threshold]

# Display customers likely to churn
print("Customers likely to churn:")
print(churned_customers)


# In[37]:


from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Creating lists for classifier names, test_accuracy_scores, f1 scores, and AUC scores.
classifier_names = []
test_accuracy_scores = []
f1_scores = []
auc_scores = []

# Plotting setup for ROC curves
plt.figure(figsize=(10, 8))

# Looping through classifiers, fitting models, and calculating train and test accuracy
for name, model in classifiers.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probability scores for the positive class
    
    classifier_names.append(name)
    
    # Calculating and storing F1 score
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)
    
    # Calculating and storing test accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    test_accuracy_scores.append(test_accuracy)

    # Calculating and storing AUC score
    auc = roc_auc_score(y_test, y_pred_proba)
    auc_scores.append(auc)
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
    
    # Printing model details
    print(f'Model: {name}')
    print(f'Training Accuracy: {accuracy_score(y_train, model.predict(X_train))}')
    print(f'Testing Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Testing AUC: {auc}')
    print('------------------------------------------------------------------')
    print(f'Testing Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
    print('------------------------------------------------------------------')
    print(f'Testing Classification report: \n{classification_report(y_test, y_pred)}')
    print('------------------------------------------------------------------')

# Plot settings for ROC Curve
plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line representing random guessing
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for All Classifiers')
plt.legend(loc='lower right')
plt.show()


# In[38]:


from sklearn.model_selection import cross_val_score

# Perform cross-validation on your model
auc_scores = cross_val_score(xgboost, X_train, y_train, cv=5, scoring='roc_auc')
print(f'Cross-validated AUC scores: {auc_scores}')
print(f'Mean AUC score: {auc_scores.mean()}')


# ### Hyperparameter Tuning

# In[39]:


print(f"Training accuracy: {model.score(X_train, y_train)}")
print(f"Test accuracy: {model.score(X_test, y_test)}")


# In[40]:


from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Define parameter grid for XGBoost
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 2]
}

grid_search = GridSearchCV(estimator=xgb.XGBClassifier(), param_grid=param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
best_xgb = grid_search.best_estimator_

# Evaluate the tuned model on the test set
print(f"Tuned Test Accuracy: {best_xgb.score(X_test, y_test)}")


# In[41]:


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Predict the probabilities for the test data
y_pred = best_xgb.predict(X_test)
y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]

# Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Print confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Calculate ROC-AUC
auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {auc}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Random classifier line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


# In[42]:


import matplotlib.pyplot as plt
import xgboost as xgb

# Plot feature importance
xgb.plot_importance(best_xgb, max_num_features=10)
plt.title("Top 10 Feature Importance")
plt.show()


# In[43]:


# Adjust the decision threshold
y_pred_adjusted = (best_xgb.predict_proba(X_test)[:, 1] > 0.4).astype(int)  # Lower threshold to 0.4

# Re-evaluate performance
print("Adjusted Classification Report:\n", classification_report(y_test, y_pred_adjusted))
print("Adjusted Confusion Matrix:\n", confusion_matrix(y_test, y_pred_adjusted))


# ### cross validation

# In[45]:


from sklearn.model_selection import cross_val_score

auc_scores = cross_val_score(best_xgb, X_train, y_train, cv=5, scoring='roc_auc')
print(f'Cross-validated AUC scores: {auc_scores}')
print(f'Mean AUC score: {auc_scores.mean()}')


# In[ ]:




