# Life Expectancy Analysis

Exploratory data analysis and data cleaning on the WHO Life Expectancy dataset, 
examining how health, economic, and social factors relate to life expectancy across countries.

## Dataset
- Source: [Life Expectancy (WHO) - Kaggle](link)
- ~2900 rows, 22 columns covering years 2000–2015 across 193 countries

## What this project does
- Loads and inspects the data (`info`, `head`, `tail`, `describe`, shape, dtypes)
- Cleans column names (strips leading/trailing whitespace)
- Checks for missing values, duplicates, and inconsistent categorical entries
- Visualizes distributions with histograms and boxplots
- Explores relationships between features and life expectancy with scatterplots
- Checks correlation and multicollinearity with a heatmap
- Handles missing values with median imputation (select columns) and `KNNImputer` 
  (applied across all numeric columns together)
- Treats outliers using IQR capping

## Tech stack
- Python (pandas, numpy)
- matplotlib, seaborn
- scikit-learn

## How to run
\```bash
pip install -r requirements.txt
python ML.py
\```

## Key findings
_(fill in once you review the plots/output — e.g. which features correlate 
most strongly with life expectancy, which columns had the most missing data)_

## Project structure
\```
├── ML.py
├── README.md
├── LICENSE
├── .gitignore
└── Life Expectancy Data.csv   (not included — see note below)
\```

## Note on the dataset
The raw CSV isn't included in this repo. Download it from the Kaggle link above 
and place it in the project root before running the script.
