# AWS Certified Machine Learning – Specialty
Learning Notes for [AWS Certified Machine Learning – Specialty](https://aws.amazon.com/certification/certified-machine-learning-specialty/?trk=8e4f8862-b963-4ec1-bad5-91e77ca9a0bb&sc_channel=ps&sc_campaign=acquisition&sc_medium=TC-P|PS-GO|Brand|Desktop|AW|Training%20and%20Certification|Certification|ASEAN|EN|Text|xx|SEM|PMO21-12347&s_kwcid=AL!4422!3!467351729653!e!!g!!machine%20learning%20aws%20certification&ef_id=CjwKCAiAlfqOBhAeEiwAYi43F8zml-JsY8NiBGt_bMUNMn0RblJBFYyk8NxyQZ1Slw8MlV9wDIDeXRoCs4gQAvD_BwE:G:s&s_kwcid=AL!4422!3!467351729653!e!!g!!machine%20learning%20aws%20certification)

<img src=https://d1.awsstatic.com/training-and-certification/Certification%20Badges/AWS-Certified_Machine-Learning_Specialty_512x512.6ac490d15fe033a3d67ca544ecd0bcbcb10d391a.png width=150>

This credential helps organizations identify and develop talent with critical skills for implementing cloud initiatives. Earning AWS Certified Machine Learning – Specialty validates expertise in building, training, tuning, and deploying machine learning (ML) models on AWS.

## Contents

* [Data Engineering](#data-engineering)
* [Exploratory Data Analysis](#exploratory-data-analysis)
* [Modeling](#modeling)
* [Algorithms](#algorithms)
* [Implementation and Operations](#implementation-and-operations)

## Data Engineering

### 1. Handling Missing Data

#### Do nothing

Let algorithm either replace missing values through imputation (XGBoost) or just ignore them (LightGBM) with ``use_missing = False`` 

#### Remove the entire record
#### Mode/median/average value replacement

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="median")
imputer = imputer.fit(train)
train = imputer.transform(train).ravel()
```

#### Most frequent value

  * Doesn't factor correlation between features
  * Works with categorical features
  * Can introduce bias
  * ``stragtegy="most_frequent"``

#### Model-based imputation

* K-Nearest Neighbors (feature similarity to predict missing values)
* Regression
* Deep Learning

#### Interpolation / extrapolation

#### Forward filling / backward filling

#### Hot deck imputation 

Randomly choosing the missing value from a set of related and similar variables

### 2. Feature Extraction and Selection

* High feature to observation ratio casues overfitting.
* Lower dimensions are easier to visualize than higher dimensions.

#### Feature selection
1. Normalization (rescales the values into a range of [0,1])
2. Remove features based on **variance thresholds**

```python
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold

normalized_data = preprocessing.normalize(data)
selector = VarianceThreshold()
selected_feature = selector.fit_transform(normalized_data)
```

#### Feature extration
1. Standardization (rescales data to have a mean (μ) of 0 and standard deviation (σ) of 1 (unit variance))
2. PCA creates new features by linearly combining original features
  * New features are uncorrelated, i.e. orthogonal
  * New features are ranked in order of "explained variance"
  * PCA can speed up machine learning while having good accuracy

```python
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)

variance_ratio = pca.explained_variance_ratio_
total_variance = variance_ratio[0] + variance_ratio[1]
```

### 3. Encoding Categorical Values

#### Binarizer encoding

```python
# select categorical features
cat_features = features_df.select_dtypes(include=["object"]).copy()

# which features have NaN value?
cat_features.columns[cat_features.isna().any()].tolist()

# replace by most frequent class
imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

from sklearn.preprocessing import LabelBinarizer
label_style = LabelBinarizer()
label_style.fit_transofrm(data)
```

#### Label encoding

Ordinal encoder 

```python
from sklearn.preprocessing import LabelEncoder
# create workclass_code
LabelEncoder().fit_transform()
```

#### One-hot-encoding

```python
pd.get_dummies(cat_features, columns=["workclass"])
```

### 4. Numerical Engineering

* Change numeric values on the same scale
  * Normalization
  * Standardization
* Binning (quantization)
  * Categorical binning (e.g. Asia - China, Singapore, Japan...)
  * Numerical binning
  * Quantile binning
```python
# bins are of different sizes, data is evenly distributed
pd.qcut(df["price"], q=4, labels=["sophomore","junior","senoir","professional"])

# bins are of equal size, data is unevenly distributed
pd.cut(df["price"], bins=4)
```

### 5. Text Feature Editing

* Bag-of-Words

Tokenize raw text and creates s statistical representataion of the text

* N-Gram

Extension of Bag-of-Words which produces groups of words of n size

* Orthogonal Space Bigram

* TF-IDF (term frequency–inverse document frequency)

<image src=https://miro.medium.com/max/1200/1*V9ac4hLVyms79jl65Ym_Bw.jpeg width=400>

### 6. AWS Migration Services and Tools


:point_up_2: [back](#contents)

## Exploratory Data Analysis

:point_up_2: [back](#contents)

## Modeling

:point_up_2: [back](#contents)

## Algorithms

:point_up_2: [back](#contents)

## Implementation and Operations

:point_up_2: [back](#contents)




