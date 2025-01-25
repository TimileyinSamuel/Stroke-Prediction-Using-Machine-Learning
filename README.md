# Stroke Prediction Using Machine Learning: A Comparative Analysis of Logistic Regression,Support Vector Machine, and Random Forest

This project explored the application of machine learning to predict stroke occurrences using a dataset comprising demographic, medical, and lifestyle data. Preprocessing steps included handling missing values, encoding categorical variables, feature scaling, and addressing severe class imbalance through SMOTE, oversampling, undersampling, and class weighting techniques.

Three models—Logistic Regression, Support Vector Machine (SVM), and Random Forest—were selected based on their unique strengths: Logistic Regression for its interpretability, SVM for its ability to handle non-linear relationships, and Random Forest for its capability to capture complex feature interactions. Exploratory Data Analysis (EDA) and data visualizations were conducted to uncover trends and insights, guiding feature selection and engineering.

The models were evaluated using performance metrics like accuracy, precision, recall, and F1-score, with a special focus on recall to address the high stakes of false negatives in healthcare. Additional improvements involved hyperparameter tuning and ensemble modeling, such as a voting strategy to leverage the strengths of all three models.

This project demonstrates the potential of machine learning to assist in early stroke detection, offering scalable and data-driven insights to support timely medical intervention.


1.0 Introduction and Background

1.1 Problem Statement
Cardiovascular diseases, including stroke, are the second leading cause of death globally, with 13.7 million cases annually (WHO, 2020). In the U.S., approximately 795,000 strokes occur each year, resulting in 140,000 deaths, with higher mortality in females (60%) than males (40%) (CDC, 2020). While 80% of strokes are preventable with early detection, they can affect all ages, with one in four occurring in individuals under 65.

1.2 Importance of Early Detection through Prediction
Early stroke prediction is critical for enhancing survival and reducing disabilities. Timely intervention, including medications, surgery, and rehabilitation, significantly improves patient outcomes.
This report examines machine learning models: Logistic Regression, Support Vector Machine, and Random Forest to predict stroke events, using a clinical dataset from Kaggle. The project aims to determine the model with the highest predictive capability, supporting early intervention to lower stroke-related mortality and disability.

1.3 Dataset Overview
This study utilizes the Kaggle Stroke Prediction Dataset, which includes various demographic, health, and lifestyle attributes to predict stroke likelihood. The dataset comprises 5,110 entries, each representing a unique patient. Key features include a unique patient ID, gender ("Male," "Female," "Other"), age (in years), and binary indicators for hypertension and heart disease history. Additional attributes cover marital status ("Yes" or "No"), work type ("Private," "Self-employed," "Govt_job," "Children," or "Never worked"), and residence type (urban or rural). The dataset also records average glucose level (mg/dL), BMI (kg/m²), and smoking status ("Formerly smoked," "Never smoked," "Smokes," "Unknown"). The target variable, stroke, indicates whether a patient has had a stroke (1 for stroke, 0 for no stroke).

1.4 Model choices and rationale
This project uses three machine learning models namely Logistic Regression, Support Vector Machine (SVM), and Random Forest to predict stroke events based on clinical and lifestyle data. Each model offers unique strengths suitable for the dataset and the requirements of stroke prediction.

1. Logistic Regression
Logistic Regression is a widely used model for binary classification and serves as an effective baseline for this study. Its primary advantages include:
•	Interpretability: Logistic Regression provides clear and interpretable coefficients, allowing healthcare professionals to identify significant risk factors such as age, hypertension, and heart disease. This transparency is critical in understanding the underlying drivers of stroke risk.
•	Baseline Performance: As a simple yet robust model, Logistic Regression offers reasonable accuracy with minimal overfitting, making it an ideal starting point for predictive modeling.
•	Class Imbalance Handling: The model can be adapted to handle imbalanced datasets through techniques like class weighting and threshold adjustment, essential given the low prevalence of stroke cases in the dataset.

2. Support Vector Machine (SVM)
Support Vector Machine is a powerful classifier, particularly effective in distinguishing between classes in complex feature spaces. Its key strengths include:
•	Non-linearity Handling: SVM leverages non-linear kernels, such as the radial basis function (RBF) kernel, to model intricate relationships between features, effectively capturing subtle interactions that influence stroke risk.
•	Class Separation: By maximizing the margin between classes, SVM ensures strong generalization, which is crucial for distinguishing between stroke and non-stroke cases in a dataset with overlapping feature distributions.
•	Suitability for Small Datasets: SVM performs exceptionally well with smaller datasets, making it an appropriate choice for this study, where the minority class (stroke cases) is limited.

3. Random Forest
Random Forest is renowned for its predictive power and ability to handle diverse datasets. It was chosen for the following reasons:
•	Feature Importance: Random Forest identifies the most influential features in the dataset, providing valuable insights for healthcare professionals to prioritize high-risk patients.
•	Complex Interactions: The model excels at capturing non-linear relationships and feature interactions, such as the combined effects of age, hypertension, smoking status or other features on stroke risk.
•	Robustness and Stability: By averaging predictions from multiple decision trees, Random Forest minimizes overfitting and delivers consistent results, particularly when working with imbalanced datasets where stroke cases are rare.







2.0 Data Exploration and Visualization
The following steps were taken to explore and visualize the dataset.
2.1 Exploratory Data Analysis (EDA)
 
To streamline the analysis, the dataset was divided into numerical and categorical columns for efficient access and organization.

![Picture 1](https://github.com/user-attachments/assets/e208dc93-d276-4030-98c5-6721f5a7d63d)




