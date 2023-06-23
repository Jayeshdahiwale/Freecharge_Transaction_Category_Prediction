# FreeCharge Transaction Category Prediction

![image](https://raw.githubusercontent.com/Jayeshdahiwale/data_for_categorization/main/freecharge-company.jpg)

With the increasing volume of financial transactions, it becomes challenging for individuals and businesses to manually categorize each transaction. By leveraging transaction category prediction, it becomes easier to automatically classify transactions into predefined categories, such as groceries, utilities, entertainment, or travel. This automation simplifies expense tracking and budgeting processes.

Transaction category prediction plays a vital role in personal finance management. By accurately categorizing transactions, individuals can gain insights into their spending patterns, identify areas where they are overspending, and make informed decisions about budgeting and saving. It allows users to track their expenses across different categories and understand their financial habits.


## Table of Content
  * [Problem Statement](#problem-statement)
  * [Objective](#objective)
  * [Dataset](#dataset)
  * [Data Pipeline](#data-pipeline)
  * [Installation](#installation)
  * [Project Structure](#project-structure)
  * [Tools Used](#tools-used)
  * [Performed Model Result](#performed-model-Result)
  * [Project Summary](#project-summary)
  * [Conclusion](#conclusion)


## Problem Statement
* We are given a data about transactions. Our goal is to categorise those transaction to their Category and SubCategories
* Total rows provided in the dataset are 53000 and 8 columns
* In the fast-paced world of banking and finance, transaction categorization has emerged as a crucial tool for banks to efficiently manage financial data and empower their customers. By automatically classifying transactions into specific categories, such as groceries, utilities, entertainment, or travel, banks can streamline financial processes, provide personalized insights, and enhance overall customer experience. This blog explores the significance of transaction categorization for banks and the benefits it offers to both the institutions and their customers.
* A study conducted by McKinsey & Company found that transaction categorization plays a crucial role in improving customer experience by providing personalized financial insights and recommendations. This leads to higher customer satisfaction and increased loyalty.




## Objective
The classification goal is to predict the Category and SubCategory of the transaction based on given features


## Dataset
The dataset is from an FreeCharge organization. The dataset provides the various transaction information takes place during the span of 2020 t0 2022. It includes over 53000 records and 8 attributes. Each attribute is a potential risk factor. These attributes are demographic, behavioral and medical risk factors. 



## Data Pipeline
1. EDA: 
    - EDA or Exploratory Data Analysis is the critical process of performing the initial investigation  on the data.  In this initial step we went to look for different features available and tried to uncover their relevance with the target variable, through this we have observed certain trends and dependencies and drawn  certain conclusions from the dataset that will be useful for further processing.
2. Data Processing: 
    - During this stage, we looked for the data types of each feature and  corrected them. After that comes the null value and outlier detection. For the null values imputation we used Mean, Median and Mode technique and for the outlier we used Capping method to handle the outliers without any loss to the data.
3. Feature Engineering: 
    - During this stage, we went on to select the most relevant  features using the chi-square test, information gain, extra trees classifier and next comes the feature scaling in order to bring down all the values in similar range. After that comes the treatment of class imbalance in the target variable that  is done using random oversampling.
4. Model Fitting and Performance Metric: 
    - Since the data is transformed to an appropriate form  therefore, we pass it to different classification models and calculate the metrics based on which we select a final model that could give us better prediction.
    
    
## Installation
This project requires python 3.6 or any other higher versions of python.
This project need software to run this python notebook "Jupyter Notebook" or "Google colab". It is highly recommended that you install the Anaconda distribution of Python or use "Google Colab" https://colab.research.google.com/, which already has most of the above packages and more included.
 

## Project Structure
```
├── README.md
├── Dataset 
│   ├── final_train_prep.xlsx
│
│
├── EDA
│   ├── Numeric & Categoric features
│   ├── Univariate Analysis
│   ├── Bivariate Analysis
│   ├── Multivariate Analysis
│   ├── Data Cleaning
│       ├── Duplicated values
│       ├── NaN/Missing values
│   ├── Treating Skewness
│   ├── Treating Outlier 
│
├── Feature Engineering
│   ├── Encoding
|       ├── Label Encoding
|       ├── One-Hot Encoding
│   ├── Handling Multicollinerity
|       ├── Correlation
|   ├── Feature Selection
|       ├── ExtraTree Classifier
|       ├── Chi-Square Test
|       ├── Iformation Gain
|   ├── Handling Class Imbalance
|       ├── Synthetic Minority Oversampling Technique (SMOTE)
│
├── Model Building
│   ├── Train Test Split
│   ├── Scaling data
│   ├── Model selection
│  
|
│   
├── Report
├── Result
└── Reference
```




## Project Summary
Importing necessary libraries and dataset. Then perform EDA to get a clear insight of the each feature, The raw data was cleaned by treating the outliers and null values. Transformation of data was done in order to ensure it fits well into machine learning models. Then finally the cleaned form of data was passed into different models and the metrics were generated to evaluate the model and then we did hyperparameter tuning in order to ensure the correct parameters being passed to the model. Then check all the model_result in order to select final model based on business application.


## Conclusion
In general, it is good practice to track multiple metrics when developing a machine learning model as each highlights different aspects of model performance. However we are dealing with Heathcare data and our data is imbalanced for that perticular reason we are more focusing towards the Recall score and F1 score.

   - Keeping into mind the sizze of dataset, applying hypeparameter tuning was a very cumbersome task and it was vey vey time taking. 
   - A single classification model is taking long to train on the data. Hence, I limit this project only on a single classification model i.e. XGBoost
   - Tried Implemneting BERT, but having the limited RAM on Google colab, we are not able to train this model.
   - Metrics For XGBoost Classification :
        - Trained the XGBoost Model with 32,000 rows and dimension 800 dimensions
        
        
        Accuracy of the model : 81%
        Precision of the model : 0.87
        Recall of the model :  0.81
        F1-score : 0.83
        - Overall the XGBoost model we can say a good fit.


***************************THANK You******************************************