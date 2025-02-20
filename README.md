﻿**STROKE PREDICTION USING MACHINE LEARNING: A COMPARATIVE ANALYSIS OF LOGISTIC REGRESSION, SUPPORT VECTOR MACHINE AND RANDOM FOREST**


This project explored the application of machine learning to predict stroke occurrences using a dataset comprising demographic, medical, and lifestyle data. Preprocessing steps included handling missing values, encoding categorical variables, feature scaling, and addressing severe class imbalance through SMOTE, oversampling, undersampling, and class weighting techniques.

Three models—Logistic Regression, Support Vector Machine (SVM), and Random Forest—were selected based on their unique strengths: Logistic Regression for its interpretability, SVM for its ability to handle non-linear relationships, and Random Forest for its capability to capture complex feature interactions. Exploratory Data Analysis (EDA) and data visualizations were conducted to uncover trends and insights, guiding feature selection and engineering.

The models were evaluated using performance metrics like accuracy, precision, recall, and F1-score, with a special focus on recall to address the high stakes of false negatives in healthcare. Additional improvements involved hyperparameter tuning and ensemble modeling, such as a voting strategy to leverage the strengths of all three models.


**1.0 Introduction and Background**

**1.1 Problem Statement**

Cardiovascular diseases, including stroke, are the second leading cause of death globally, with 13.7 million cases annually (WHO, 2020). In the U.S., approximately 795,000 strokes occur each year, resulting in 140,000 deaths, with higher mortality in females (60%) than males (40%) (CDC, 2020). While 80% of strokes are preventable with early detection, they can affect all ages, with one in four occurring in individuals under 65.

**1.2 Importance of Early Detection through Prediction**

Early stroke prediction is critical for enhancing survival and reducing disabilities. Timely intervention, including medications, surgery, and rehabilitation, significantly improves patient outcomes.

This report examines machine learning models: Logistic Regression, Support Vector Machine, and Random Forest to predict stroke events, using a clinical dataset from Kaggle. The project aims to determine the model with the highest predictive capability, supporting early intervention to lower stroke-related mortality and disability.

**1.3 Dataset Overview**

This study utilizes the Kaggle Stroke Prediction Dataset, which includes various demographic, health, and lifestyle attributes to predict stroke likelihood. The dataset comprises 5,110 entries, each representing a unique patient. Key features include a unique patient ID, gender ("Male," "Female," "Other"), age (in years), and binary indicators for hypertension and heart disease history. Additional attributes cover marital status ("Yes" or "No"), work type ("Private," "Self-employed," "Govt\_job," "Children," or "Never worked"), and residence type (urban or rural). The dataset also records average glucose level (mg/dL), BMI (kg/m²), and smoking status ("Formerly smoked," "Never smoked," "Smokes," "Unknown"). The target variable, stroke, indicates whether a patient has had a stroke (1 for stroke, 0 for no stroke).

**1.4 Model choices and rationale**

This project uses three machine learning models namely Logistic Regression, Support Vector Machine (SVM), and Random Forest to predict stroke events based on clinical and lifestyle data. Each model offers unique strengths suitable for the dataset and the requirements of stroke prediction.

**1. Logistic Regression**

Logistic Regression is a widely used model for binary classification and serves as an effective baseline for this study. Its primary advantages include:

- **Interpretability**: Logistic Regression provides clear and interpretable coefficients, allowing healthcare professionals to identify significant risk factors such as age, hypertension, and heart disease. This transparency is critical in understanding the underlying drivers of stroke risk.
- **Baseline Performance**: As a simple yet robust model, Logistic Regression offers reasonable accuracy with minimal overfitting, making it an ideal starting point for predictive modeling.
- **Class Imbalance Handling**: The model can be adapted to handle imbalanced datasets through techniques like class weighting and threshold adjustment, essential given the low prevalence of stroke cases in the dataset.

**2. Support Vector Machine (SVM)**

Support Vector Machine is a powerful classifier, particularly effective in distinguishing between classes in complex feature spaces. Its key strengths include:

- **Non-linearity Handling**: SVM leverages non-linear kernels, such as the radial basis function (RBF) kernel, to model intricate relationships between features, effectively capturing subtle interactions that influence stroke risk.
- **Class Separation**: By maximizing the margin between classes, SVM ensures strong generalization, which is crucial for distinguishing between stroke and non-stroke cases in a dataset with overlapping feature distributions.
- **Suitability for Small Datasets**: SVM performs exceptionally well with smaller datasets, making it an appropriate choice for this study, where the minority class (stroke cases) is limited.

**3. Random Forest**

Random Forest is renowned for its predictive power and ability to handle diverse datasets. It was chosen for the following reasons:

- **Feature Importance**: Random Forest identifies the most influential features in the dataset, providing valuable insights for healthcare professionals to prioritize high-risk patients.
- **Complex Interactions**: The model excels at capturing non-linear relationships and feature interactions, such as the combined effects of age, hypertension, smoking status or other features on stroke risk.
- **Robustness and Stability**: By averaging predictions from multiple decision trees, Random Forest minimizes overfitting and delivers consistent results, particularly when working with imbalanced datasets where stroke cases are rare.







**2.0 Data Exploration and Visualization**

The following steps were taken to explore and visualize the dataset.

**2.1 Exploratory Data Analysis (EDA)**

![Picture 1](https://github.com/user-attachments/assets/70b12aad-4ab0-4af4-af0c-ca2cbda69ee1)


To streamline the analysis, the dataset was divided into numerical and categorical columns for efficient access and organization.

![Picture 1](https://github.com/user-attachments/assets/028c28e3-9e63-4f8a-9918-661c1a942cfe)



- **Categorical Data**: The categorical columns were examined to identify unique values and their frequencies. For example, the *gender* column includes three unique values: "Female" (2,994 entries), "Male" (2,115 entries), and "Other" (1 entry).
- **Numerical Data**: Key statistics such as mean, median, and range were calculated for numerical columns. Notably, the *BMI* column has 3.9% missing values (201 entries). Rather than removing rows with missing values, which would reduce the dataset size, the missing BMI values were imputed with the mean.

The dataset was also examined to understand the distribution of the target variable, *stroke*. It was observed that 95.1% of patients (4,817 entries) did not experience a stroke, while 4.87% (249 entries) did. This class imbalance highlights the need for evaluation metrics that account for minority class performance, such as recall and F1-score.




**2.2 Visualization**

Both interactive (Plotly, Cufflinks) and static (Seaborn, Matplotlib) libraries were used to generate various plots, including histograms, bar charts, pie charts, and box plots, to visualize data patterns and detect outliers.

**Bar Plot for Categorical Features**:

![Picture 1](https://github.com/user-attachments/assets/0911bb3a-0d35-41cd-b56e-c5906556f89e)


Bar plots reveal key categorical feature distributions. More females (3,000) than males (2,100) are present, which could indicate gender disparities in stroke risk. Marital status, with most patients married, may indirectly affect stroke risk through social and economic factors, particularly when combined with health-related features.

The *work type* plot shows that most patients fall into the "Private" category, followed by "Self-employed" and "Govt\_job," with smaller counts in the "Children" and "Never worked" groups. This distribution may reflect differences in stress levels, healthcare access, and lifestyle habits, all of which influence stroke risk. Lower counts in "Children" and "Never worked" likely represent younger or less exposed populations with naturally lower stroke risk.

The *residence type* plot displays a nearly equal split between urban and rural residents, enabling balanced analysis. Variations in stroke risk between urban and rural populations could arise from differences in healthcare access, environmental factors, or lifestyle choices.

The *smoking status* plot shows that most individuals have "Never smoked," followed by "Unknown," "Formerly smoked," and "Smokes." Smoking, a significant stroke risk factor, is notably absent in most of the population, but those who smoke or have smoked are at higher risk. The "Unknown" category introduces some uncertainty but remains relevant for modeling when combined with features like age, BMI, and heart disease.

**Box Plot for Numerical Features**:

![Picture 1](https://github.com/user-attachments/assets/42902d9a-5b62-45f9-8901-e0d0512ed6d2)


The box plots provide insights into the distribution and variability of numerical features, highlighting key factors influencing stroke risk. The *age* distribution is symmetric, with values ranging from infancy to around 80 years and a median near 45. This balanced representation allows the model to assess stroke prevalence across age groups, a critical risk factor.

The *avg\_glucose\_level* plot is skewed, with most values below 150 mg/dL and several outliers exceeding 200 mg/dL, suggesting the presence of metabolic conditions like diabetes, which elevate stroke risk. Similarly, the *BMI* distribution is skewed, with most values between 20 and 40 kg/m² but extreme outliers nearing 100 kg/m², indicative of severe obesity.

In summary, the box plots reveal that age is normally distributed, while glucose levels and BMI display skewness with notable outliers, pointing to potential health risks that could impact stroke prediction.



**Distribution Plots for Numerical Features**:

![Picture 1](https://github.com/user-attachments/assets/2bbfee64-374b-4d89-8e55-9f60b1f28ec3)



The distribution plots provide valuable insights into the spread and central tendencies of key features, revealing patterns and anomalies that inform data preprocessing and modeling.

The *age* distribution is symmetric, with a mean of 43.23 years and a median of 45 years. This broad age range and balanced representation around middle age are beneficial for capturing age-related variations in stroke risk across younger and older populations.

The *avg\_glucose\_level* plot is right-skewed, with a mean of 106.14 mg/dL and a median of 91.88 mg/dL. Most values are below 150 mg/dL, but outliers exceeding 200 mg/dL suggest metabolic conditions like diabetes, a known stroke risk factor.

The *BMI* distribution is also right-skewed, with a mean of 27.76 kg/m² and a median of 27.7 kg/m². Most values fall between 20 and 40, but extreme outliers nearing 100 kg/m² indicate severe obesity, often associated with hypertension and diabetes. This highlights BMI’s potential as a critical predictor of stroke risk.


**Stroke Rates by Work Type and Residence Type**

![Picture 1](https://github.com/user-attachments/assets/a7b860f7-cd6b-41a5-8872-6fa344df2915)



The heatmap illustrates stroke rates by work type and residence type, highlighting the distribution of stroke (1) and non-stroke (0) cases. The intensity of color represents the count of individuals in each category, with darker shades indicating higher counts.

Among work types, the "Private" sector shows the highest stroke occurrence, with 68 rural and 81 urban stroke cases out of over 1,300 individuals in each group. This may reflect lifestyle factors such as stress, long hours, or limited healthcare access, particularly for urban workers, who experience slightly higher stroke rates. "Self-employed" individuals show moderate stroke cases, with 31 rural and 34 urban cases, indicating minimal variation between rural and urban residents. This suggests that residence type may have less impact on stroke risk for this group.

Government workers have relatively low stroke rates, with 14 rural and 19 urban cases, potentially due to job stability or better healthcare access, which may reduce stroke risk. Similarly, the "Children" and "Never worked" categories show minimal stroke cases (just one in each category) likely due to younger ages or fewer exposure to common stroke risk factors.

The heatmap also reveals a subtle increase in stroke cases among urban residents, particularly in the "Private" and "Self-employed" categories. This suggests that urban lifestyles, potentially involving higher stress and environmental factors, may contribute to stroke risk.



**3.0 Data preprocessing:**

The following preprocessing stages were applied to ensure model performance and improve predictive accuracy:

**3.1 Data Cleaning**

- **Handling Missing Values**: The BMI column had 3.9% missing values (201 entries). To retain as much data as possible, missing values were imputed using the mean BMI. This approach was chosen to preserve the dataset's size and distribution while minimizing bias, as dropping rows would reduce the dataset significantly.
- **Dropping Erroneous Entries**: The gender column included one "Other" entry, which was considered an outlier and inconsistent with the binary classification of gender in the dataset. This single instance was removed to maintain data consistency and avoid introducing skew in gender-based analysis. The id column was also dropped as it is irrelevant for the models.
- **Data Splitting and Organization**: The dataset was split into training (77%) and testing (23%) sets using scikit-learn's train\_test\_split function. This ratio ensured sufficient data for model training while preserving a separate test set for unbiased evaluation. Predictor variables were separated from the target variable (stroke) to streamline the modeling process.
- **Feature Scaling**: Numerical features were standardized using scikit-learn’s StandardScaler to normalize data, particularly benefiting distance-based models like SVMs. While scaling showed minimal impact on accuracy for some models, it was retained for consistency across all algorithms and to prevent issues with scale-sensitive models.

**3.2 Outlier Detection and Handling**

Outliers in numerical features can distort the model’s learning process, especially in healthcare data where extreme values might indicate data anomalies rather than typical patient conditions. Techniques such as capping or binning were considered; in this case, large and highly variable numerical features like BMI and glucose level were binned into three or four categories. Binning outliers into defined ranges helps reduce their influence while retaining valuable information, which is critical in a healthcare setting where extreme values can signify potential health risks.

**3.3 Encoding Categorical Variables**

Categorical features, including gender, smoking status, work type, marital status, and residence type, were encoded into numerical values to make them compatible with machine learning algorithms. Label encoding was applied to binary features (e.g., marital status and residence type), while one-hot encoding was applied to multi-category variables (e.g., work type and smoking status). This encoding ensures that the model can effectively interpret categorical data without imposing an ordinal relationship where none exists. Label encoding and one-hot encoding were both implemented using scikit-learn’s preprocessing functions to maintain consistency.


**4.0 Model Evaluation Metrics**

The performance of each model was evaluated using a set of metrics: accuracy, precision, recall, and F1-score. Each metric offers unique insights into the model's classification capabilities, particularly crucial for predicting a rare event like stroke. In this context, certain metrics are prioritized over others due to the high stakes of false positives and false negatives in healthcare.

1. **Recall:** Recall, or sensitivity, is the most critical metric in stroke prediction. It measures the proportion of actual stroke cases that are correctly identified by the model. In healthcare, false negatives (i.e., missing a stroke case) can be life-threatening, as patients left undiagnosed may miss timely interventions that could prevent severe disability or death. Therefore, maximizing recall is essential to ensure that as many true stroke cases as possible are correctly detected, reducing the likelihood of missed diagnoses.
1. **Precision:** Precision is the second most important metric in this context, as it measures the proportion of true positives among all predicted positive cases. High precision reduces the rate of false positives, which, while less critical than false negatives, can still have significant implications. A high number of false positives could lead to unnecessary medical interventions, anxiety, and resource expenditure. In the stroke prediction context, high precision ensures that patients flagged as at risk for stroke are genuinely at high risk, supporting efficient and effective healthcare resource allocation.
1. **F1-Score:** The F1-score, the harmonic mean of precision and recall, serves as a balanced metric that considers both false positives and false negatives. Given the importance of both precision and recall in stroke prediction, the F1-score is particularly valuable in assessing the model’s overall performance. It provides a single, balanced metric that ensures the model not only captures as many stroke cases as possible (high recall) but also maintains accuracy in its positive predictions (high precision).
1. **Accuracy:** Accuracy is the least critical metric in this context because it measures the overall correctness of the model’s predictions, including both positive and negative cases. Since stroke events are rare, a model could achieve high accuracy by primarily predicting the majority class (non-stroke), which would provide a misleadingly optimistic view of model performance. While accuracy gives a general overview, it is not as informative in the context of class imbalance, where recall, precision, and F1-score better capture the model’s effectiveness in identifying stroke cases.

**5.0 Model Training and Evaluation**

<a name="_hlk180094632"></a>This section outlines the approach and results from developing models to predict stroke occurrence based on medical data. 

Three different machine learning models which are Logistic Regression, Support Vector Machine (SVM), Random Forest all were implemented and evaluated. All the three model were evaluated once with a defined function for easier computing.

The goal was to accurately predict stroke risk by testing several techniques for model improvement, including **scaling, class balancing, feature engineering, oversampling,** **under sampling**, **SMOTE (Synthetic Minority Over-sampling Technique)**. Ultimately, the model performances were compared, and insights were drawn based on their effectiveness, highlighting the best-performing models for each metric.

The respective result is as follows:

**5.1 Initial Model Performance**

![Picture 1](https://github.com/user-attachments/assets/4f420e4e-b2a3-41cd-95d7-689e5621c228)



The initial performance of the models demonstrates high accuracy but significant challenges in detecting stroke cases due to class imbalance. All three models: Logistic Regression, Random Forest, and SVM achieved an accuracy of 94.31%. However, they failed to identify any stroke cases, as evidenced by zero values for precision, recall, and F1-score. This indicates that all models are defaulting to predicting the majority class (non-stroke), neglecting the minority class entirely.

These results emphasize the need for techniques to address the class imbalance and improve the detection of stroke cases.









**5.2 Five-fold cross-validation**

![Picture 1](https://github.com/user-attachments/assets/774587f5-eec7-4851-88b1-e218af8ed4a8)



The application of 5-fold cross-validation provided more stable accuracy measures but had minimal impact on stroke detection. Logistic Regression showed an accuracy increase from 94.31% to 95.11%, with a precision of 0.3333. However, recall remained extremely low at 0.0040, resulting in an F1-score of 0.0079, indicating persistent difficulty in identifying stroke cases.

Random Forest maintained a high accuracy of 94.98% but struggled with stroke detection. Precision dropped to 0.0120, recall remained low at 0.0153, and the F1-score was just 0.0153. While cross-validation stabilized the accuracy, the model continued to miss most stroke cases.

SVM had a slight accuracy increase to 95.13%, but precision, recall, and F1-score remained at 0.000, indicating no improvement in detecting strokes and a complete reliance on predicting the majority class.

In summary, cross-validation improved model stability but failed to address the key issue of class imbalance and stroke detection. Additional techniques such as resampling, feature engineering, or hyperparameter tuning are needed to improve performance on the minority class.

**5.3. Class Imbalance Issues and Handling for the three models** 

The dataset suffers from significant class imbalance, with far fewer stroke cases than non-stroke cases. This imbalance causes models to favor predicting the majority class, leading to high overall accuracy but poor detection of strokes, which are critical in this context.

1. **Class Weights**

To address this, class weights were introduced to the models, assigning higher weights to the minority class (stroke). This adjustment encourages the models to focus more on correctly identifying stroke cases, balancing the influence of each class during training. Class weighting is a simple yet effective approach, particularly in healthcare, where detecting rare events like strokes is crucial for improving patient outcomes.

![Picture 1](https://github.com/user-attachments/assets/3682c5b6-b467-43be-bcb9-a68ef0ecc665)



The introduction of class weights led to improvements in stroke detection for some models, particularly in recall, while others saw minimal or no improvement. Logistic Regression saw accuracy drop to 71.88%, but recall increased significantly to 0.7708, alongside gains in precision (0.1401) and F1-score (0.2376), indicating a better balance in identifying stroke cases.

Random Forest maintained high accuracy at 94.30%, but precision, recall, and F1-score remained at 0.000, showing no improvement and a continued inability to detect any stroke cases.

SVM demonstrated notable progress, with recall rising to 0.7500 and modest increases in precision (0.1465) and F1-score (0.2439), although its accuracy dropped to 73.46%. This indicates a better trade-off between identifying strokes and overall performance.

This suggests that class weights may not be sufficient for Random Forest and require complementary approaches to address the class imbalance effectively.




1. **Undersampling**

<a name="_hlk180097678"></a>**Undersampling** was applied to address the class imbalance by reducing the majority class (non-stroke cases) to match the size of the minority class (stroke cases). This technique creates a balanced dataset by removing excess non-stroke samples, enabling the models to better focus on distinguishing between the two classes. The table below presents the evaluation metrics after applying under sampling. 

![Picture 1](https://github.com/user-attachments/assets/1758bf35-025b-435f-9980-fe4d67b47340)



Applying undersampling significantly improved stroke detection across all models, with notable gains in recall and F1-score. Logistic Regression achieved an accuracy of 76.47%, with recall increasing to 0.8039 and F1-score rising to 0.7756, reflecting a strong improvement in balancing predictions for stroke and non-stroke cases.

Random Forest showed marked improvement, with accuracy at 76.15%, recall increasing to 0.8363, and an F1-score of 0.7667. These results highlight its enhanced ability to detect stroke cases while maintaining a good balance with precision.

SVM also benefited significantly, achieving an accuracy of 76.14%, recall of 0.8232, and the highest F1-score among the models at 0.7771, indicating strong performance in balancing precision and recall.

In summary, undersampling proved highly effective in enhancing stroke detection, especially for Random Forest and SVM. Although accuracy decreased slightly due to the balanced dataset, the improved recall and F1-scores demonstrate that the models are now far better at identifying stroke cases, addressing the prior imbalance effectively.

1. **SMOTE (Synthetic Minority Over-sampling Technique)** 

SMOTE** was applied to address the class imbalance by generating synthetic samples for the minority class (stroke cases). This approach balances the dataset without reducing the number of majority class samples, allowing the model to learn more effectively from a balanced dataset while preserving the original data distribution. The table below presents the evaluation metrics after applying SMOTE.

![Picture 1](https://github.com/user-attachments/assets/8eaa2db5-b5a6-4f84-a07c-19aa89e0671d)


Applying SMOTE significantly improved the models' stroke detection performance, with notable gains in precision, recall, and F1-score across all models. Logistic Regression achieved an accuracy of 80.38%, with precision at 0.7916, recall at 0.8247, and an F1-score of 0.8077. These metrics indicate a much-improved balance in identifying both stroke and non-stroke cases.

Random Forest emerged as the best-performing model, achieving an accuracy of 94.22%. It recorded a precision of 0.9254, recall of 0.9617, and an F1-score of 0.9439, demonstrating exceptional effectiveness in detecting strokes while maintaining strong performance in non-stroke predictions.

SVM also showed improvements, with an accuracy of 77.39%. It achieved a precision of 0.7441, recall of 0.8340, and an F1-score of 0.7869, reflecting a more balanced performance in stroke prediction.

In summary, SMOTE effectively addressed the class imbalance, significantly enhancing the models' stroke detection capabilities.

1. **Oversampling**

Oversampling was applied to address the class imbalance by replicating minority class (stroke) samples to create a balanced dataset. This technique increases the representation of stroke cases without altering the majority class, allowing the models to better learn patterns associated with the minority class. The table below presents the evaluation metrics for each model after applying oversampling.

![Picture 1](https://github.com/user-attachments/assets/363fc219-4b43-4dcb-9199-c6a6b5d51535)


Applying oversampling improved model performance across all metrics, particularly in precision, recall, and F1-score. Logistic Regression achieved an accuracy of 76.83%, with precision of 0.7547, recall of 0.7953, and an F1-score of 0.7744. These results indicate a substantial improvement in the model's ability to detect strokes while maintaining a balance between precision and recall.

Random Forest performed exceptionally well, achieving an accuracy of 99.24%. It reached near-perfect metrics, with precision of 0.9840, recall of 1.000, and an F1-score of 0.9921. This makes Random Forest the most effective model for stroke prediction, excelling in both minority and majority class detection.

SVM also improved, achieving an accuracy of 75.80%, precision of 0.7359, recall of 0.8056, and an F1-score of 0.7690. While showing better balance in predictions, SVM continues to lag behind Logistic Regression and Random Forest in terms of overall performance.

**5.3. Feature Engineering and Hyperparameter Tuning for model improvement**

To enhance prediction quality, new features were derived from medical, demographic, and lifestyle data, enriching the models with more relevant information. Key engineered features include:

- **cv\_risk\_score**: A cardiovascular risk score based on hypertension, heart disease, and BMI.
- **metabolic\_syndrome**: A flag for high-risk metabolic syndrome cases.
- **smoking\_cv\_risk**: A combined risk factor using smoking status and cardiovascular indicators.
- **Glucose Levels and BMI Categories:** Continuous variables like glucose level and BMI were categorized into ranges to capture risk levels associated with different ranges (e.g., normal, elevated, and high levels). Binning these variables can help the model identify risk thresholds and improve interpretability.
- **Age Categories:** Age was also binned into categories (e.g., young, middle-aged, senior) to account for age-related risk factors. Feature engineering these variables allows the model to capture medically relevant patterns, potentially improving accuracy.

![Picture 1](https://github.com/user-attachments/assets/2ae5a6f7-e22a-41ff-9728-71ba882e2ec7)


The application of feature engineering introduced new medically relevant features to the dataset, aiming to improve the models’ ability to detect stroke cases. However, the results indicate mixed outcomes in terms of prediction performance.

Logistic Regression achieved an accuracy of 95.18%, slightly higher than previous results. Precision increased to 0.4167, but recall dropped to 0.0281, resulting in an F1-score of 0.0518. This indicates that while the model has improved its precision for positive predictions, it struggles significantly in identifying actual stroke cases, as reflected in the low recall.

Random Forest showed an accuracy of 94.08%, with precision at 0.0667 and recall at 0.0040, resulting in a very low F1-score of 0.0078. These results suggest that despite the introduction of more features, Random Forest was unable to leverage the engineered variables to improve its stroke detection performance meaningfully. It continues to suffer from a severe imbalance in its ability to predict the minority class.

SVM demonstrated no improvement in stroke detection, with precision, recall, and F1-score remaining at 0.000, even though the accuracy was slightly higher at 95.12%. This shows that the additional features did not aid the SVM model in addressing the class imbalance, as it still predicts only the majority class.






- **Hyperparameter Tuning using Grid Search optimized each model by fine-tuning parameters for improved performance.**

![Picture 1](https://github.com/user-attachments/assets/443e1962-0623-41aa-a1a6-e85ddeb71067)


  Grid search result integration



![Picture 1](https://github.com/user-attachments/assets/3e585074-584c-4a6b-8c72-98d144494b1e)



After hyperparameter tuning, the performance of Random Forest and Support Vector Machine (SVM) showed some improvement in detecting stroke cases, while Logistic Regression continued to struggle.

Logistic Regression maintained an accuracy of 95.18%, with precision at 0.4167. However, recall remained low at 0.0281, resulting in a poor F1-score of 0.0518. This indicates that while Logistic Regression improved slightly in terms of precision, it still fails to effectively identify stroke cases, as reflected in its low recall.

Random Forest achieved an accuracy of 95.01%, with precision at 0.0833, recall at 0.0080, and an F1-score of 0.0231. Although accuracy remains high, the model’s ability to identify stroke cases remains limited, showing minimal improvement in recall and F1-score, which are critical for detecting minority class instances.

SVM maintained an accuracy of 95.12%, but its precision, recall, and F1-score remained at 0.000, indicating no improvement in its ability to detect stroke cases. Despite hyperparameter tuning, the SVM model still fails to address the class imbalance effectively, consistently predicting only the majority class.

- **Dropping of redundant features**

![Picture 1](https://github.com/user-attachments/assets/87e12871-aeda-4445-adee-ba17d4fcf6e6)


**5.4.** A *Voting Strategy* was also implemented, combining predictions from Logistic Regression, Random Forest, and SVM. This ensemble approach leverages the strengths of each model, enhancing accuracy and consistency to create a robust stroke prediction system.

![Picture 1](https://github.com/user-attachments/assets/9d19eb1e-12ec-4b3e-84c7-b5c511f8764a)


6. **Result Discussion and Conclusion**

![Picture 1](https://github.com/user-attachments/assets/61d6db01-3620-49cf-8f0e-83f0febd8da6)


The final results highlight the performance of Logistic Regression, Random Forest, and Support Vector Machine (SVM) in predicting stroke cases, with evaluations based on precision, recall, F1-score, and accuracy.

**Logistic Regression** excels in non-stroke prediction (Class 0) with an F1-score of 0.98 but fails to detect strokes (Class 1), scoring 0.00 for precision, recall, and F1. Its 95% accuracy highlights the majority class bias, reflected by low macro average precision and recall (0.48).

**Random Forest** shows more balanced performance compared to the other models. For Class 0, it achieves a precision of 0.96, recall of 0.95, and an F1-score of 0.95. For Class 1 (stroke cases), it outperforms Logistic Regression and SVM, with a precision of 0.18, recall of 0.21, and F1-score of 0.19. Although these scores for stroke detection remain low, they represent an improvement in minority class prediction. Its overall accuracy of 91% and macro average precision and recall of 0.57 and 0.58, respectively, highlight its better handling of class imbalance. However, the limited recall for stroke cases indicates room for further improvement.

**Support Vector Machine (SVM)** performs similarly to Logistic Regression. It excels in non-stroke prediction (Class 0) with a precision of 0.95, recall of 1.00, and an F1-score of 0.98. Like Logistic Regression, it fails to detect any stroke cases (Class 1), with precision, recall, and F1-score all at 0.00. Its overall accuracy of 95% reflects strong non-stroke detection, but macro average precision (0.48) and recall (0.50) indicate poor performance for the minority class.

**6.1 Confusion Matrices**

**Confusion matrices** were generated for each model to visualize how well the models predicted stroke occurrences, displaying the number of true positives, true negatives, false positives, and false negatives.

![Picture 1](https://github.com/user-attachments/assets/43129ce2-4ea2-4bc9-85b0-edb7a630cf01)


The confusion matrices illustrate each model's ability to predict stroke (Class 1) and non-stroke (Class 0) cases. Logistic Regression and SVM both perform well in identifying non-stroke cases, achieving high true negatives, as evident from the large number of correctly classified non-stroke cases. However, both models fail entirely in detecting stroke cases, with zero true positives and a high number of false negatives. This indicates that Logistic Regression and SVM are biased toward the majority class, predicting nearly all cases as non-stroke.

Random Forest, on the other hand, shows some capability in identifying stroke cases, correctly classifying 10 true positives while still missing many (false negatives). Although it continues to prioritize non-stroke cases, it demonstrates a better balance than Logistic Regression and SVM by identifying a small number of stroke cases.

**6.2 Comparison and Best Model Assessment**

While all models excel in predicting non-stroke cases (Class 0) with high precision and recall, Random Forest outshines the others by achieving the highest metrics for stroke case detection (Class 1). With a precision of 0.18, recall of 0.21, and an F1-score of 0.19 for Class 1, Random Forest demonstrates an improved ability to balance predictions across both classes. This makes it the most reliable model for handling the class imbalance inherent in the dataset.

In contrast, Logistic Regression and SVM fail to detect any stroke cases, with zero precision, recall, and F1-score for the minority class. Their high overall accuracy stems solely from correctly predicting the majority class (non-stroke), underscoring their ineffectiveness in addressing the critical task of stroke detection.

In conclusion, Random Forest emerges as the strongest candidate for stroke prediction due to its ability to balance performance across both classes. Despite its limitations in recall and F1-score for stroke cases, it surpasses Logistic Regression and SVM in identifying the minority class, making it the most effective model for this application. Further improvements could involve advanced techniques to enhance stroke case detection further.

**6.3 Why Logistic Regression and SVM Struggled with Stroke Prediction**

Logistic Regression and Support Vector Machine (SVM) struggled with stroke detection due to the dataset's severe class imbalance, with over 95% of cases being non-strokes. Despite applying class weights, both models remained biased toward the majority class.

- Logistic Regression: Assumes linear relationships, which are insufficient to capture the complex interactions in stroke prediction. Even with class weights, the scarcity of stroke cases limits its ability to identify meaningful decision boundaries.
- SVM: Focuses on maximizing the margin between classes, which is dominated by the majority class in imbalanced datasets. Class weights and limited stroke cases fail to improve its ability to distinguish minority class patterns.

**7. Conclusion**

This project aimed to predict stroke occurrences using Logistic Regression, Random Forest, and Support Vector Machine (SVM). While all models performed well in predicting non-stroke cases, their ability to detect strokes varied significantly due to the dataset's class imbalance.

Logistic Regression and SVM achieved high accuracy for non-stroke predictions but completely failed to identify stroke cases, making them unsuitable for balanced stroke detection. Random Forest demonstrated the best overall performance, achieving a better balance by identifying more stroke cases, as confirmed by the confusion matrix analysis.

In conclusion, Random Forest emerged as the most reliable model for stroke prediction, effectively balancing stroke and non-stroke detection.
