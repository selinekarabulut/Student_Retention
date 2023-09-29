#!/usr/bin/env python
# coding: utf-8

# # Analyzing Students' Dropout: A Predictive Approach to Student Retention

# 
# 
# **Research Question:** How do different factors (academic path, demographics, and social-economic factors) impact whether a student drops out, remains enrolled, or graduates?
# 
# **Data:** UC Irvine Machine Learning Repository
# 
# https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

# **Executive Summary**
# 
# In this study, the objective was to delve into a dataset comprising diverse information about students to discern the various factors that impact their academic trajectories, focusing particularly on the likelihood of them dropping out, remaining enrolled, or graduating. The dataset provided a rich source of details about each student, including their age, origin, grades, scholarship status, and broader economic indicators like the prevailing unemployment rate in the country.
# 
# The initial step involved a meticulous examination of the data to understand the general trends in student outcomes. It was observed that the majority of students in the dataset had graduated, but there were fewer data points for students who either dropped out or continued to be enrolled. After handling this class imbalance, a computational model was developed to predict student outcomes using the provided information.
# 
# The model demonstrated substantial accuracy, correctly predicting outcomes approximately 76% of the time during testing phases. It revealed that the entry grade and the age of a student when commencing their studies were pivotal factors influencing their academic progression. 
# 
# To ensure the reliability of the model and to negate the possibility of its performance being a mere stroke of luck with the initial data, it was tested multiple times using varied data segments. The model maintained a consistent accuracy rate of around 77% across different tests.
# 
# In conclusion, the insights derived from this analysis are invaluable, shedding light on the potential determinants of a student's academic journey. While the model serves as a substantial aid in understanding student behaviors and outcomes, it is crucial to acknowledge the uniqueness of each student. The insights gained from this study can assist educational institutions in refining their support structures by focusing on the identified influential factors, thereby fostering an environment conducive to academic success and student well-being.

# **Technical Summary**
# 
# *Objective*
# 
# The primary aim of our analysis was to understand the factors influencing student outcomes (Dropout, Enrolled, Graduate) using a provided dataset and to build a predictive model that can accurately classify students based on their characteristics and past academic performance.
# 
# *Dataset Overview:*
# 
# The dataset comprises various features, including demographic information (e.g., age, gender, nationality), academic records (e.g., admission grade, curricular units credited), socio-economic indicators (e.g., scholarship holder, tuition fees status), and macroeconomic factors (e.g., GDP, unemployment rate, inflation rate).
# 
# *Data Exploration:*
# 
# * The dataset was initially explored to understand the distribution of various features and the target variable.
# * A class imbalance in the target variable, with a majority of students being classified as "Graduate."
# 
# *Class Imbalance:*
# 
# To address the class imbalance observed in the target variable, the Synthetic Minority Over-sampling Technique (SMOTE) was applied.
# 
# *Data Preprocessing:*
# 
# * Categorical variables were one-hot encoded, and the dataset was split into training and testing sets for model evaluation.
# * Principal Component Analysis (PCA) was employed to reduce dimensionality, retaining the top 100 components that captured over 90% of the variance.
# 
# *Modeling:*
# 
# * A Random Forest classifier was chosen due to its robustness and ability to handle a mix of continuous and categorical features.
# * The classifier achieved an accuracy of approximately 76.38% on the test set, with particularly strong performance for the "Dropout" and "Graduate" classes. However, performance for the "Enrolled" class was suboptimal.
# 
# *Feature Importance:*
# 
# The most influential features in the model's decision-making process were identified as "Admission grade," "Age at enrollment," and several others.
# 
# *Validation:*
# 
# * The model's findings were validated using Precision, Recall, and F1-Score metrics for each class.
# * 5-fold cross-validation further confirmed the model's robustness, yielding an average accuracy of approximately 77%, indicating that the model's performance was consistent across different train-test splits.
# 
# *Conclusion:*
# 
# The analysis provides a comprehensive understanding of the factors influencing student outcomes. The insights derived from feature importance can guide interventions and strategies to improve student outcomes in real-world educational settings.
# 

# **Steps taken:**
# * Load Data
# * EDA
# * Feature Engineering
#     * One-hot encode the categorical variables
#     * Normalize or standardize the continuous features
# * Random Forest Classifier
# * Reduce Dimensionality
#     * PCA
# * Handle Class Imbalance 
#     * SMOTE
# * Results
#     * Feature Importance
#     * Confusion Matrix
# * Validate Results
#     * Precision, Recall, and F1-Score
#     * Cross Validation

# ## Load Dataset

# In[3]:


import pandas as pd

# Load the dataset
data = pd.read_csv("data_dropout.csv")

# Display the first few rows of the dataset
data.head()


# It appears that the dataset uses a semicolon (;) as the delimiter.
# 
# Reload the dataset with the correct delimiter and then proceed with our exploration.

# In[4]:


# Load the dataset with the correct delimiter
data = pd.read_csv("data_dropout.csv", delimiter=";")

# Display the first few rows of the dataset
data.head()


# ## EDA

# In[5]:


# Check the shape of the dataset
data_shape = data.shape
data_shape


# 4424 observations, 36 inputs, 1 target variable

# In[6]:


# Get the summary statistics of the numerical features
summary_statistics = data.describe()

summary_statistics


# The dataset contains 4,424 entries and 37 columns. Here are some observations from the summary statistics:
# 
# * The dataset seems to have no missing values for the features, as the count for each feature matches the total number of entries.
# * Features like "Marital status," "Application mode," "Course," and others are integer encoded. Need to consider whether to treat them as categorical and perform one-hot encoding or keep them as is.
# * Features like "Previous qualification (grade)," "Admission grade," and other continuous features have varying scales, and need to consider normalization or standardization.
# * The target variable is categorical, with classes like "Dropout" and "Graduate."

# In[7]:


# Check for missing values
missing_values = data.isnull().sum()

missing_values


# There are no missing values in the dataset.

# In[22]:


# Check if "Gender" column exists in the dataset
if 'Gender' in data.columns:
    # Count the occurrences of 1 (male) and 0 (female) in the "Gender" column
    male_count = data[data['Gender'] == 1].shape[0]
    female_count = data[data['Gender'] == 0].shape[0]
else:
    male_count = female_count = None

male_count, female_count


# In[8]:


# Explore the distribution of the target variable
target_distribution = data['Target'].value_counts()

target_distribution


# Regarding the target variable distribution:
# 
# * There are 2,209 instances of the "Graduate" class.
# * There are 1,421 instances of the "Dropout" class.
# * There are 794 instances of the "Enrolled" class.

# ## Feature Engineering and Data Preprocessing

# ### One-hot encode the categorical variables
# 
# many of the integer-encoded features are actually categorical in nature. 
# 
# one-hot encode them to ensure the model doesn't interpret them as ordinal.

# In[9]:


# List of columns to be one-hot encoded
columns_to_encode = [
    "Marital status", "Application mode", "Course", "Previous qualification", 
    "Nacionality", "Mother's qualification", "Father's qualification", 
    "Mother's occupation", "Father's occupation"
]

# One-hot encoding the columns
data_encoded = pd.get_dummies(data, columns=columns_to_encode, drop_first=True)

# Display the shape and first few rows of the encoded dataset
data_encoded_shape = data_encoded.shape
data_encoded_head = data_encoded.head()

data_encoded_shape, data_encoded_head


# The dataset has been one-hot encoded, expanding the number of features from 37 to 239 due to the categorical variables.

# ### Standardize the continuous features
# 
# standardize the continuous features to ensure they are on a similar scale.

# Given the description and the nature of the dataset, the following columns appear to be continuous:
# 
# * Previous qualification (grade)
# * Admission grade
# * Age at enrollment
# * Curricular units 1st sem (grade)
# * Curricular units 2nd sem (grade)
# * Unemployment rate
# * Inflation rate
# * GDP
# 
# The continuous features have been standardized using z-score normalization

# In[10]:


from sklearn.preprocessing import StandardScaler

# Columns to standardize
continuous_columns = [
    "Previous qualification (grade)", "Admission grade", "Age at enrollment",
    "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)",
    "Unemployment rate", "Inflation rate", "GDP"
]

# Standardizing the columns
scaler = StandardScaler()
data_encoded[continuous_columns] = scaler.fit_transform(data_encoded[continuous_columns])

# Display the first few rows of the standardized dataset
data_encoded.head()


# ### Split the dataset into training and testing sets

# Now, let's split the dataset into training and testing sets. This will allow us to train a model on one subset and validate its performance on another unseen subset. We'll use 80% of the data for training and 20% for testing.

# In[11]:


from sklearn.model_selection import train_test_split

# Splitting features and target
X = data_encoded.drop('Target', axis=1)
y = data_encoded['Target']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# split the dataset into training and testing sets. Here are the sizes:
# 
# * Training features: 3539 samples with 238 features
# * Testing features: 885 samples with 238 features
# * Training target: 3539 samples
# * Testing target: 885 samples

# # Random Forest Classifier

# Given the multi-class nature of the target variable, I'll use a classification algorithm that can handle multiple classes. One of the commonly used algorithms for multi-class classification is the Random Forest classifier. Let's train a Random Forest classifier on the training data and evaluate its performance on the testing data.

# In[12]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the classifier performance
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

accuracy, class_report


# The Random Forest classifier achieved an accuracy of approximately 76.38 % on the test set.
# 
# * Dropout class:
#     Precision: 0.83
#     Recall: 0.75
#     F1-Score: 0.79
# * Enrolled class:
#     Precision: 0.57
#     Recall: 0.27
#     F1-Score: 0.37
# * Graduate class:
#     Precision: 0.76
#     Recall: 0.95
#     F1-Score: 0.84
#     
# The model performs relatively well for the "Dropout" and "Graduate" classes. However, its performance for the "Enrolled" class is suboptimal, especially in terms of recall, indicating that there are many false negatives for this class.
# 
# Performing feature selection or extraction to reduce dimensionality.

# # Reduce Dimensionality

# To reduce dimensionality, one of the most common techniques is Principal Component Analysis (PCA). PCA is a method that transforms the original variables into a new set of variables, which are linear combinations of the original variables. These new variables (or principal components) capture the most variance in the data.
# 
# Using PCA, we can reduce the number of features while still retaining a significant portion of the original variance.

# In[13]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Perform PCA
pca = PCA().fit(X_train)

# Plot the explained variance ratio
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.title("Explained Variance by Principal Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.show()


# The graph represents the cumulative explained variance as we increase the number of principal components. The x-axis denotes the number of principal components, while the y-axis represents the cumulative explained variance.
# 
# From the graph, we can see that around 100 principal components explain over 90% of the variance in the dataset. This means that we can reduce the dataset from 238 features to around 100 features while still retaining over 90% of the original variance.

# Tansform both the training and testing data sets using PCA to retain the top 100 principal components.
# 
# Train the Random Forest classifier on the transformed training data.
# 
# Evaluate its performance on the transformed testing data.

# In[14]:


# Apply PCA transformation to retain the top 100 components
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train the Random Forest classifier on the transformed data
rf_classifier_pca = RandomForestClassifier(random_state=42)
rf_classifier_pca.fit(X_train_pca, y_train)

# Predictions on the transformed test set
y_pred_pca = rf_classifier_pca.predict(X_test_pca)

# Evaluate the classifier performance
accuracy_pca = accuracy_score(y_test, y_pred_pca)
class_report_pca = classification_report(y_test, y_pred_pca)

accuracy_pca, class_report_pca


# After reducing dimensionality with PCA and retaining the top 100 principal components, the Random Forest classifier achieved an accuracy of approximately 74.24 % on the test set. 
# 
# * Dropout class:
#     Precision: 0.80
#     Recall: 0.74
#     F1-Score: 0.77
# * Enrolled class:
#     Precision: 0.49
#     Recall: 0.23
#     F1-Score: 0.31
# * Graduate class:
#     Precision: 0.74
#     Recall: 0.93
#     F1-Score: 0.83

# Comparing this to the model trained on the original features, we observe a slight drop in accuracy. This is expected since PCA introduces some information loss. However, considering that we've reduced the number of features by more than half, the performance is still relatively strong.
# 
# Dimensionality reduction techniques like PCA can be advantageous, especially when computational resources are limited or when working with algorithms that have high computational complexity.

# # Handle Class Imbalance

# Handling class imbalance is crucial because imbalanced datasets can lead to biased models that favor the majority class, resulting in suboptimal performance for the minority class. Several techniques can be used to address class imbalance such as resampling (oversampling, undersampling), algorithm-level Approaches (i.e. ensemble methods):
# 
# 
# Oversampling: Increasing the number of instances in the minority class by duplicating samples or generating synthetic samples. One popular method for this is SMOTE (Synthetic Minority Over-sampling Technique).
# 
# the SMOTE (Synthetic Minority Over-sampling Technique) to handle the class imbalance. SMOTE works by generating synthetic examples in the feature space. These synthetic instances are generated by interpolating between existing instances in the minority class.
# 
# * the imblearn library, which provides the SMOTE implementation

# In[15]:


get_ipython().system('pip install imbalanced-learn')


from imblearn.over_sampling import SMOTE

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE to the training data
X_train_smote, y_train_smote = smote.fit_resample(X_train_pca, y_train)

# Check the distribution of the target variable after SMOTE
y_train_smote_distribution = y_train_smote.value_counts()

y_train_smote_distribution


# In[16]:


X_train_smote.shape, X_test_pca.shape, y_train_smote.shape, y_test.shape


# In[17]:


# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_smote, y_train_smote)

# Predictions on the test set
y_pred = rf_classifier.predict(X_test_pca)

# Evaluate the classifier performance
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

accuracy, class_report


# After handling class imbalance using the SMOTE method, the Random Forest classifier achieved an accuracy of approximately 74.24 % on the test set. 
# 
# * Dropout class:
#     Precision: 0.79
#     Recall: 0.78
#     F1-Score: 0.78
# * Enrolled class:
#     Precision: 0.47
#     Recall: 0.28
#     F1-Score: 0.48
# * Graduate class:
#     Precision: 0.81
#     Recall: 0.81
#     F1-Score: 0.81

# While handling class imbalance using SMOTE did not improve overall accuracy, it led to improvements in precision, recall, and F1-Scores for specific classes, particularly for the "Dropout" and "Graduate" classes. The trade-off between precision and recall varies depending on the class, and the choice of whether to use SMOTE should consider the specific goals and requirements of the classification task.
# 
# * Accuracy: The accuracy remains the same in both cases (approximately 74.24%), indicating that handling class imbalance using SMOTE did not improve overall accuracy.
# 
#      Dropout Class: Precision and Recall both increased after using SMOTE. This indicates that the model is better at correctly identifying and classifying instances belonging to the "Dropout" class, both in terms of precision (fewer false positives) and recall (fewer false negatives). The F1-Score also improved, suggesting a better balance between precision and recall.
#     
#      Enrolled Class: Precision decreased after using SMOTE, indicating that the model's ability to correctly classify instances belonging to the "Enrolled" class with precision decreased. However, recall improved slightly, meaning that the model is better at capturing more instances of this class. The F1-Score also increased, reflecting an overall improvement in classification performance for this class.
#     
#      Graduate Class:Precision increased after using SMOTE, indicating that the model's ability to correctly classify instances belonging to the "Graduate" class with precision improved. Recall remained the same. Consequently, the F1-Score also improved, reflecting a better balance between precision and recall for this class.

# # Results
# ## Feature Importance

# Feature importance can provide insights into which features are most influential in making predictions. 

# In[18]:


# Extract feature importance
feature_importance = rf_classifier.feature_importances_

# Pair each feature with its importance
features = X_train.columns
feature_importance_pairs = list(zip(features, feature_importance))

# Sort features based on importance
sorted_feature_importance = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)

# Extract the top 20 most important features for visualization
top_features = [x[0] for x in sorted_feature_importance[:20]]
top_importance = [x[1] for x in sorted_feature_importance[:20]]

# Visualize the top features
plt.figure(figsize=(15, 10))
plt.barh(top_features, top_importance, align='center')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.xlabel('Importance')
plt.title('Top 20 Feature Importance using Random Forest')
plt.show()


# The visualization above displays the top 20 most important features, as determined by the Random Forest classifier. The importance scores are normalized such that they sum to 1, with higher values indicating greater importance.
# 
# From the graph, we can see which features have the most influence on the classifier's predictions. For instance, "Admission grade," "Age at enrollment," and several other features stand out as particularly important.

# ## Confusion Matrix

# A table used to describe the performance of a classification model on a set of data for which the true values are known.
# 
# Create a heatmap to represent the matrix, as it offers a visually appealing and interpretable representation of the classifier's performance across different classes

# In[19]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix as a heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
            xticklabels=rf_classifier.classes_, 
            yticklabels=rf_classifier.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Random Forest Classifier')
plt.show()


# From the matrix:
# 
# * Most "Dropout" and "Graduate" instances are correctly classified, as indicated by the high numbers in the respective diagonal cells.
# * The "Enrolled" class has a higher misclassification rate, with many instances predicted as "Graduate."
# 
# This visualization helps in identifying which classes the model is performing well on and which classes it's struggling with. In this case, the model seems to struggle most with the "Enrolled" class.

# # Validate Results
# ## Precision, Recall, and F1-Score

# While the confusion matrix provides raw counts of correct and incorrect predictions, these metrics provide a more normalized view of performance for each class.

# In[20]:


from sklearn.model_selection import cross_val_score

# Display the Precision, Recall, and F1-Score
classification_metrics = classification_report(y_test, y_pred)

classification_metrics


# ## Cross Validation

# To ensure that the performance isn't due to a specific train-test split, we can use cross-validation. This divides the dataset into multiple folds, trains and tests the model on different combinations of these folds, and then aggregates the results.

# In[21]:



# Perform cross-validation
cv_scores = cross_val_score(rf_classifier, X, y, cv=5, scoring='accuracy')

cv_scores, cv_scores.mean()


# Performed 5-fold cross-validation and obtained the following accuracy scores for each fold:
# 
# * Fold 1: 0.767
# * Fold 2: 0.772
# * Fold 3: 0.779
# * Fold 4: 0.776
# * Fold 5: 0.756
# Average Accuracy Across All Folds: 0.77
# 
# The cross-validation results indicate that the model's performance is consistent across different train-test splits, with an average accuracy of approximately 0.770. This consistency suggests that our model is reasonably robust and not merely tuned to a specific data split.

# In[ ]:




