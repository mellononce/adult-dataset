import sns
from matplotlib import pyplot as plt
from pandas import Categorical
from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Example: Encoding categorical variables and handling missing values
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
# Example: Chi-square test for independence (replace 'race' and 'income' with relevant columns)
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

# Data Cleaning
# Check for missing values
missing_values = X.isnull().sum()
print(missing_values[missing_values > 0])

# Example: Filling missing values
# For numerical columns
X.loc[:, 'age'] = X['age'].fillna(X['age'].mean())
X.loc[:, 'fnlwgt'] = X['fnlwgt'].fillna(X['fnlwgt'].mean())
X.loc[:, 'education-num'] = X['education-num'].fillna(X['education-num'].mean())
X.loc[:, 'capital-gain'] = X['capital-gain'].fillna(X['capital-gain'].mean())
X.loc[:, 'capital-loss'] = X['capital-loss'].fillna(X['capital-loss'].mean())
X.loc[:, 'hours-per-week'] = X['hours-per-week'].fillna(X['hours-per-week'].mean())

# For categorical columns
X.loc[:, 'sex'] = X['sex'].fillna(X['sex'].mode()[0])
X.loc[:, 'workclass'] = X['workclass'].fillna(X['workclass'].mode()[0])
X.loc[:, 'education'] = X['education'].fillna(X['education'].mode()[0])
X.loc[:, 'relationship'] = X['relationship'].fillna(X['relationship'].mode()[0])
X.loc[:, 'race'] = X['race'].fillna(X['race'].mode()[0])
X.loc[:, 'native-country'] = X['native-country'].fillna(X['native-country'].mode()[0])



# Convert data types
#X['some_categorical_column'] = X['some_categorical_column'].astype('category')
#X['some_numerical_column'] = X['some_numerical_column'].astype(float)

sns.boxplot(x=X['age'])
plt.show()

sns.boxplot(x=X['fnlwgt'])
sns.boxplot(x=X['education-num'])
sns.boxplot(x=X['capital-gain'])
sns.boxplot(x=X['capital-loss'])
sns.boxplot(x=X['hours-per-week'])




# Standardization
#scaler = StandardScaler()
#X_standardized = pd.DataFrame(scaler.fit_transform(X[numerical_vars]), columns=numerical_vars)

# Normalization
#scaler = MinMaxScaler()
#X_normalized = pd.DataFrame(scaler.fit_transform(X[numerical_vars]), columns=numerical_vars)

# One-Hot Encoding
#X_encoded = pd.get_dummies(X[categorical_vars])

X.info()
X.describe()


# EDA
print(X.describe())
sns.countplot(x='education', data=X)
plt.show()

# Calculate correlation matrix for numeric columns only
numeric_cols = X.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_cols.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()


# More plots and analysis
# Countplots for other categorical variables
categorical_vars = ['workclass', 'marital-status', 'occupation', 'race', 'sex']
for var in categorical_vars:
    sns.countplot(x=var, data=X)
    plt.xticks(rotation=45)
    plt.show()

# Histograms for numerical variables
numerical_vars = ['age', 'hours-per-week', 'capital-gain', 'capital-loss']
for var in numerical_vars:
    X[var].hist(bins=20)
    plt.title(var)
    plt.show()

# Boxplots
for var in numerical_vars:
    sns.boxplot(x='income', y=var, data=X)
    plt.show()

# Pairplot for numerical variables
sns.pairplot(X[numerical_vars])
plt.show()

# Bar charts for categorical variables vs income
for var in categorical_vars:
    sns.barplot(x=var, y='income', data=X)  # Adjust if 'income' is not directly in X
    plt.xticks(rotation=45)
    plt.show()

# Correlation matrix
corr_matrix = X[numerical_vars].corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()

# FacetGrid example
g = sns.FacetGrid(X, col="sex", row="race", margin_titles=True)
g.map(plt.hist, "age", bins=np.linspace(10, 80, 15))

# Violin plots
for var in numerical_vars:
    sns.violinplot(x='income', y=var, data=X)  # Adjust if 'income' is not directly in X
    plt.show()

# Example: Chi-square test for independence (replace 'race' and 'income' with relevant columns)
from scipy.stats import chi2_contingency

contingency_table = pd.crosstab(X['race'], X['income'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square test result: chi2 = {chi2}, p-value = {p}")

# Example: Encoding categorical variables and handling missing values
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Encoding categorical variables
encoder = OneHotEncoder(sparse=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_vars]))
X_encoded.columns = encoder.get_feature_names(categorical_vars)

# Handling missing values
imputer = SimpleImputer(strategy='mean')
X_numerical = pd.DataFrame(imputer.fit_transform(X[numerical_vars]))
X_numerical.columns = numerical_vars

# Combine encoded and numerical data
X_preprocessed = pd.concat([X_numerical, X_encoded], axis=1)


