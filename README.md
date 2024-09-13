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

### Identifying and Removing outliers
![Screenshot 2024-09-09 204816](https://github.com/user-attachments/assets/a7bae8ed-ab26-4182-81ce-922b8b01a8e9)
![Screenshot 2024-09-09 204828](https://github.com/user-attachments/assets/4f6689db-5791-4236-b6ef-18c8f9768584)

- Strip plot for a zoomed in view of outliers (Blood Pressure, Height and Weight)
![Screenshot 2024-09-11 094217](https://github.com/user-attachments/assets/c5cb745b-f9cd-49c7-85fe-74ea5bc9e284)
![Screenshot 2024-09-11 094229](https://github.com/user-attachments/assets/98a6c5cf-8104-41c7-9e5e-e75a561a5a59)

- After removal of outliers
![Screenshot 2024-09-11 094239](https://github.com/user-attachments/assets/fadec8e3-df35-4746-9f00-277e7a7e8ded)
![Screenshot 2024-09-11 094254](https://github.com/user-attachments/assets/b7b3edb5-7729-49a2-9115-aba05205e84b)




### Dataset Visualization
- Histogram features
![Screenshot 2024-09-02 205343](https://github.com/user-attachments/assets/da2e3dfe-3175-4709-a78d-c17fe1f1a27c)

- Heatmap
![Screenshot 2024-09-02 205325](https://github.com/user-attachments/assets/341702bd-9bc7-4815-8c5f-0b4922bbe2b9)



