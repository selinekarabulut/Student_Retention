# Student_Retention
Analyzing Students' Dropout: A Predictive Approach to Student Retention

Objective
The primary aim of our analysis was to understand the factors influencing student outcomes (Dropout, Enrolled, Graduate) using a provided dataset and to build a predictive model that can accurately classify students based on their characteristics and past academic performance.

Dataset Overview:
The dataset comprises various features, including demographic information (e.g., age, gender, nationality), academic records (e.g., admission grade, curricular units credited), socio-economic indicators (e.g., scholarship holder, tuition fees status), and macroeconomic factors (e.g., GDP, unemployment rate, inflation rate).

Data Exploration:
The dataset was initially explored to understand the distribution of various features and the target variable.
A class imbalance in the target variable, with a majority of students being classified as "Graduate."

Class Imbalance:
To address the class imbalance observed in the target variable, the Synthetic Minority Over-sampling Technique (SMOTE) was applied.

Data Preprocessing:
Categorical variables were one-hot encoded, and the dataset was split into training and testing sets for model evaluation.
Principal Component Analysis (PCA) was employed to reduce dimensionality, retaining the top 100 components that captured over 90% of the variance.

Modeling:
A Random Forest classifier was chosen due to its robustness and ability to handle a mix of continuous and categorical features.
The classifier achieved an accuracy of approximately 76.38% on the test set, with particularly strong performance for the "Dropout" and "Graduate" classes. However, performance for the "Enrolled" class was suboptimal.

Feature Importance:
The most influential features in the model's decision-making process were identified as "Admission grade," "Age at enrollment," and several others.

Validation:
The model's findings were validated using Precision, Recall, and F1-Score metrics for each class.
5-fold cross-validation further confirmed the model's robustness, yielding an average accuracy of approximately 77%, indicating that the model's performance was consistent across different train-test splits.

Conclusion:
The analysis provides a comprehensive understanding of the factors influencing student outcomes. The insights derived from feature importance can guide interventions and strategies to improve student outcomes in real-world educational settings.

https://public.tableau.com/app/profile/selin8335/viz/AnalyzingStudentsDropout/Dashboard1

