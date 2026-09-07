import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
 
data = pd.read_csv("Life Expectancy Data.csv")
data.columns = data.columns.str.strip()  # remove leading/trailing whitespace from column names
 
# to analyse the data
print(data.info())
 
# To view the first five
print(data.head())
 
# to view all columns
print(data.columns)
 
# to view the last five
print(data.tail())
 
# statistical view of data
print(data.describe())
 
# shape of data
print(data.shape)
 
# finding missing values
missing_values = data.isnull().sum()
print(missing_values)
 
# missing values in percentage
percent_missing_values = missing_values / data.shape[0] * 100
print(percent_missing_values)
 
# checking the duplicates
duplicated = data.duplicated().sum()
print("duplicated:", duplicated)
 
# finding garbage values
for i in data.select_dtypes(include="object").columns:
    print(data[i].value_counts())
    print("***" * 10)
 
# descriptive statistics
print("Descriptive statistics")
print(data.describe().T)
print(data.describe(include='object'))
 
# to visualize my data on histogram
for i in data.select_dtypes(include="number").columns:
    sns.histplot(data=data, x=i)
    plt.show()
 
# Boxplot to identify outliers
for i in data.select_dtypes(include="number").columns:
    sns.boxplot(data=data, x=i)
    plt.show()
 
# to visualize on a scatter plot
columns = data.select_dtypes(include="number").columns
print(columns)
 
for i in ['Year', 'Life expectancy', 'Adult Mortality', 'infant deaths', 'Alcohol',
          'percentage expenditure', 'Hepatitis B', 'Measles', 'BMI',
          'under-five deaths', 'Polio', 'Total expenditure', 'Diphtheria',
          'HIV/AIDS', 'GDP', 'Population', 'thinness 1-19 years',
          'thinness 5-9 years', 'Income composition of resources', 'Schooling']:
    sns.scatterplot(data=data, x=i, y='Life expectancy')
    plt.show()
 
# correlation with heatmap to interpret the relation and multi-co-linearity
corr = data.select_dtypes(include="number").corr()
print(corr)
plt.figure(figsize=(15, 15))
print(sns.heatmap(corr, annot=True))
 
# Missing value treatments
# choose method of imputing values like mean, median, mode, or KNN imputer
print("missing values:")
print(data.isnull().sum())
 
warnings.filterwarnings("ignore")
 
# median imputation for a couple of key columns
for i in ['BMI', 'Income composition of resources']:
    data[i].fillna(data[i].median(), inplace=True)
    print("Results after imputing with median")
 
# IMPUTING USING KNNIMPUTER
# Fit across all numeric columns together so KNN can actually use other
# features to find nearest neighbors (fitting one column at a time defeats
# the purpose of KNN imputation).
impute = KNNImputer()
numeric_cols = data.select_dtypes(include="number").columns
data[numeric_cols] = impute.fit_transform(data[numeric_cols])
print("Results after imputing with KNNImputer")
print(data.isnull().sum())
 
# Outlier Treatment (IQR Capping)
outlier_cols = ["GDP", "Total expenditure", "Measles", "BMI", "under-five deaths",
                 "HIV/AIDS", "Adult Mortality", "Income composition of resources",
                 "thinness 1-19 years", "thinness 5-9 years", "Schooling"]
 
for i in outlier_cols:
    q1 = data[i].quantile(0.25)
    q3 = data[i].quantile(0.75)
    IQR = q3 - q1  # Interquartile range
    LW = q1 - 1.5 * IQR
    UW = q3 + 1.5 * IQR
    n_out = ((data[i] < LW) | (data[i] > UW)).sum()
    data[i] = np.where(data[i] < LW, LW, data[i])
    data[i] = np.where(data[i] > UW, UW, data[i])
 
    print(i, "-> lower:", round(LW, 2), "upper:", round(UW, 2), "capped:", n_out)
 