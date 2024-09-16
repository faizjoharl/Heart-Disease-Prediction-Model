### Problem Statement:

Heart disease remains one of the leading causes of mortality worldwide, posing a significant burden on healthcare systems and affecting millions of lives. Early detection and accurate prediction of heart disease are critical for improving patient outcomes through timely intervention and preventive measures.

---

### Role and Objective:

As a junior data analyst, The objective of this project is to design and implement a robust machine learning model that accurately predicts the likelihood of heart disease in individuals. By analyzing a comprehensive dataset that includes clinical, demographic, and lifestyle factors, the model aims to identify key predictors and provide actionable insights.


In this machine capstone, I have collected the dataset from kaggle (https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) and I will be using a number of machine learning algorithims to find which prediction model works best in prediciting whether a person is suffering from heart disease or not.

---

### Potential Applications:

1. Early Detection & Prevention
2. Risk Assessment & Stratification
3. Treatment Optimization
4. Resource Allocation
5. Research & Clinical Trials

---

### Dataset source and features:

The dataset comprises a total of 70000 patients with age ranging from 30 to 65 years old.

Before data cleaning, the dataset 34979 records of patients with heart disease and 
35021 records of patients without heart disease.  
The "disease" column serves as a class label to categorize patients 
into two groups: those with or without heart disease.

In the original dataset presented here there are 24470 male patients, 
and 45530 female patients. The features are as follows,

	1. [id] - The unique ID of the patient
	2. [age] - The age of the patient (days)
	3. [gender] - The gender of the patient
	4. [height] - Height in cm
	5. [weight] - weight in kg
	6. [ap_hi] - systolic blood pressure
	7. [ap_lo] - diastolic blood pressure
	8. [cholesterol] - cholesterol level 1:Normal, 2: Above Normal, 3: Well Above Normal
	9. [gluc] - glucose level of patient. 1:Normal, 2: Above Normal, 3: Well Above Normal
	10.[smoke] - whether patient smokes or not
	11.[alco] - whether patient drinks or not
	12.[active] = whether patient exercises or not
	13.[disease] - Presence of disease (TARGET VARIABLE)
     
---


### Libraries

- pandas
- numpy
- matplotlib
- sklearn

---

### Data Cleaning
- Load the dataset.
- Add headers.
- Standardizing date column to a consistent format
- Convert age (days) to age (years)
- Feature engineering a BMI column
- Identifying outliers through visualization

---

### Identifying and Removing outliers
![Screenshot 2024-09-09 204816](https://github.com/user-attachments/assets/a7bae8ed-ab26-4182-81ce-922b8b01a8e9)
![Screenshot 2024-09-09 204828](https://github.com/user-attachments/assets/4f6689db-5791-4236-b6ef-18c8f9768584)

- Strip plot for a zoomed in view of outliers (Blood Pressure, Height and Weight)
![Screenshot 2024-09-11 094217](https://github.com/user-attachments/assets/c5cb745b-f9cd-49c7-85fe-74ea5bc9e284)
![Screenshot 2024-09-11 094229](https://github.com/user-attachments/assets/98a6c5cf-8104-41c7-9e5e-e75a561a5a59)

- After removal of outliers
![Screenshot 2024-09-11 094239](https://github.com/user-attachments/assets/fadec8e3-df35-4746-9f00-277e7a7e8ded)
![Screenshot 2024-09-11 094254](https://github.com/user-attachments/assets/b7b3edb5-7729-49a2-9115-aba05205e84b)

---

### Dataset Visualization
- Histogram features
![Screenshot 2024-09-02 205343](https://github.com/user-attachments/assets/da2e3dfe-3175-4709-a78d-c17fe1f1a27c)

---

- Heatmap
![Screenshot 2024-09-02 205325](https://github.com/user-attachments/assets/341702bd-9bc7-4815-8c5f-0b4922bbe2b9)

There is multicollinearity between :
‘ap_hi' and ‘ap_lo’ = 0.71,
‘height' and ‘gender’ = 0.52,
‘disease’ and ‘ap_hi’ = 0.43

Target Variable: The variable ‘disease’ does not show a strong correlation with any other features in the dataset, despite having data from 62,505 patients. This suggests that no single feature is a strong predictor of the disease on its own.

---

- Train Test Split - 1000 epoch with and without EarlyStopping & Dropout Layer

![Screenshot 2024-09-13 214428](https://github.com/user-attachments/assets/580ac615-2529-421a-8066-779d7b1d0634)
Insight:
The training accuracy is much higher than the validation accuracy, it indicate overfitting. model becomes too specific to the training data and doesn’t generalize well.

---

### Confusion Matrix and Classification Reports
![Screenshot 2024-09-16 115359](https://github.com/user-attachments/assets/5729a7c1-ad1d-441e-9c3c-c0a691cd5881)

- Compare Model Scores

![Screenshot 2024-09-16 115706](https://github.com/user-attachments/assets/1e269e09-1bbf-4761-8094-d6a70645512e)


---



### Compare ROC (Receiver Operating Characteristic) graphs
![Screenshot 2024-09-05 2155367](https://github.com/user-attachments/assets/30c2ae0a-7f0f-47f8-a065-96c28c0c5cac)
![Screenshot 2024-09-05 215631](https://github.com/user-attachments/assets/7db0b667-3eae-442a-add4-3d690c08c1e1)
![Screenshot 2024-09-05 215536](https://github.com/user-attachments/assets/30dfaca7-150a-410f-a044-0c1ad5206be1)

- Compare AUC Curve Score

![Screenshot 2024-09-13 103007](https://github.com/user-attachments/assets/e77cac92-a15d-4906-a64b-b4718d71745f)

---

### Insights

- Outliers and Data Cleaning:
Despite having a large dataset with 70,000 observations, significant outliers skewed the analysis.
After removing these outliers (leaving 62,505 observations) and ensuring a balanced distribution of heart disease cases, the prediction model still performs well.

- Goal: Capturing All Positive Cases (Recall):
When selecting a model, the focus is on the Recall score, also known as sensitivity or true positive rate (High sensitivity ensures that true positives (actual cases) are captured, minimizing missed diagnoses.).
The Random Forest model achieves the best recall at 69.73%.

- AUC and ROC:
The Logistic Regression model has the highest AUC score (0.78). AUC indicates how well the model discriminates between positive and negative cases.
AUC provides a concise summary of model performance, making it easy for clinicians to make informed decisions, and ensure accurate heart disease risk assessment..



### Challenges

- Scarcity of Large Discrete Datasets:
Obtaining large, well-labeled datasets specifically for heart disease prediction can be challenging. 
Researchers often rely on existing medical databases, which may not cover all relevant patient populations or may lack diversity. Example is this current dataset.

- Lack of valuable data features:
Although basic health assessment data are available, there is still a need for more advanced data features to get a more effective model performance.

- Feature Selection and Engineering
Choosing relevant features (variables) that contribute to accurate predictions is critical. However, identifying the most informative features can be complex. 
Feature engineering involves transforming raw data into meaningful features, but it requires domain knowledge and creativity.


### Recommendations

- Missing Vital Information:
Despite having an extensive dataset, crucial information related to heart disease is missing.
Biomarkers like Troponin levels, C-Reactive Protein (CRP), and Creatine phosphokinase (CPK) are essential for accurate predictions of heart disease

- Column Representation:
The current representation of features like “Cholesterol” and “Glucose” (using values 1, 2, or 3) lacks clarity.
Whole values or more informative categories would enhance the model’s usability.

- Better Cholesterol Representation:
Using low-density lipoprotein (LDL) cholesterol as a feature would provide a more accurate heart disease risk assessment.


In summary, incorporating relevant features and improving feature
representation can enhance the model’s accuracy and clinical relevance. 

---


