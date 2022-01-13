# AWS Certified Machine Learning – Specialty
Learning Notes for [AWS Certified Machine Learning – Specialty](https://aws.amazon.com/certification/certified-machine-learning-specialty/?trk=8e4f8862-b963-4ec1-bad5-91e77ca9a0bb&sc_channel=ps&sc_campaign=acquisition&sc_medium=TC-P|PS-GO|Brand|Desktop|AW|Training%20and%20Certification|Certification|ASEAN|EN|Text|xx|SEM|PMO21-12347&s_kwcid=AL!4422!3!467351729653!e!!g!!machine%20learning%20aws%20certification&ef_id=CjwKCAiAlfqOBhAeEiwAYi43F8zml-JsY8NiBGt_bMUNMn0RblJBFYyk8NxyQZ1Slw8MlV9wDIDeXRoCs4gQAvD_BwE:G:s&s_kwcid=AL!4422!3!467351729653!e!!g!!machine%20learning%20aws%20certification)

<img src=https://d1.awsstatic.com/training-and-certification/Certification%20Badges/AWS-Certified_Machine-Learning_Specialty_512x512.6ac490d15fe033a3d67ca544ecd0bcbcb10d391a.png width=150/>

This credential helps organizations identify and develop talent with critical skills for implementing cloud initiatives. Earning AWS Certified Machine Learning – Specialty validates expertise in building, training, tuning, and deploying machine learning (ML) models on AWS.

## Contents

* [Data Engineering](#data-engineering)
  * Handling Missing Data
  * Feature Extraction and Selection
  * Encoding Categorical Values
  * Numerical Engineering
  * Text Feature Editing
  * AWS Migration Services and Tools
* [Exploratory Data Analysis](#exploratory-data-analysis)
  * Kinesis Data Streams
  * Kinesis Data Firehose
  * Kinesis Video Streams
  * Kinesis Data Analytics
  * AWS Glue
  * Analyze and visualize data
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
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split()

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

```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,3))
counts = vectorizer.fit_transform(corpus)
```

* Orthogonal Sparse Bigram

OSBs are generated by sliding the window of size n over the text, and outputting every pair of words that includes the first word in the window.

``` 
"The quick brown fox", {The_quick, The__brown, The___fox}

"quick brown fox jumps", {quick_brown, quick__fox, quick___jumps}

"brown fox jumps over", {brown_fox, brown__jumps, brown___over}

"fox jumps over the", {fox_jumps, fox__over, fox___the}

"jumps over the lazy", {jumps_over, jumps__the, jumps___lazy}

"over the lazy dog", {over_the, over__lazy, over___dog}

"the lazy dog", {the_lazy, the__dog}

"lazy dog", {lazy_dog}
```

* TF-IDF (term frequency–inverse document frequency)

<image src=https://miro.medium.com/max/1200/1*V9ac4hLVyms79jl65Ym_Bw.jpeg width=400/>


### 6. AWS Migration Services and Tools

#### Amazon Data Pipeline

* Copy data using Pipeline Activities
* Schedule regular data movement and data processing activities
* Integrates with on-premise and cloud-based storage systems

<p align="center">
<image src=https://acg-wordpress-content-production.s3.us-west-2.amazonaws.com/app/uploads/2017/02/1_BVb4nDfOxMt2JDylmMWbIQ.png width=400/>
<p/>

#### AWS Database Migration Services (DMS)

* Move data between databases (e.g. MySQL to MySQL, Aurora to DynamoDB)

#### AWS Glue

* Extract, transform, and load (ETL) service
  * Determine data type and schema
* Can run your data engineering algorithms
  * Feature selection
  * Data cleansing
* Can run on demand, on a schedule, or events

#### Amazon SageMaker

* Use jupyter notebooks
  * Scikit-learn
  * Pandas

#### Amazon Athena

* Run SQL queries on S3 data
* Need a data catalog such as the one created by Glue
* SQL transform your data in preparation for use in ML models

#### Use Cases

Move data: 
* DMS: EMR Cluster --> S3
* Glue: DynamoDB --> S3
* Data Pipeline/ Glue: Redshift --> S3
* DMS: on-premise database --> S3

> **Amazon EMR** (previously called Amazon Elastic MapReduce) is a managed cluster platform that simplifies running big data frameworks, such as Apache Hadoop and Apache Spark, on AWS to process and analyze vast amounts of data.

> **Amazon Redshift** is a fully managed, petabyte-scale data warehouse service in the cloud.

:point_up_2: [back](#contents)

## Exploratory Data Analysis

### 1. Kinesis Data Streams

* Get data from data producers such as IoT, social media
* Use *shards* to stream data to consumers such as EC2, lambda, Kinesis Data Analytics, EMR clusters
* Consumers then send data to a data repository such as S3, DynamoDB, Redshift or BI tools

<p align="center">
<image src=./data_stream.JPG width=700/>
<p/>

> A **shard** is the base throughput unit of a Kinesis Data Stream. Data producers assign partition keys to records. Partition keys ultimately determine which shard ingests the data record for a stream.

> **Data Stream** is a logical grouping of shards, will retain data for 24 hours, or up to 7 days with extended retention enabled.

* Kinesis Producer Library

Put data into a Kinesis data stream

* Kinesis Agent

Pre-built Java application that collects and sends data to Kinesis stream. It can be installed on web servers, log servers and database servers.

<p align="center">
<image src=./karc.JPG width=500/>
<p/>

* **Key points**
  * Shards are append-only logs
  * Shards contain ordered sequence of records ordered by arrival time
  * One shard can ingest up to 1000 data records per second, or 1MB/sec
  * Specify the number of shards needed when you create a stream
  * Add/remove shards from stream dynamically as throughput changes via API, Lambda, auto scaling
  * Enhanced fan-out: one shard for each consumer
  * Non-enhanced fan-out: one shard shared across consumers

### 2. Kinesis Data Firehose

* Get data from data producers such as IoT, social media
* Use *Lambda* functioning instead of shards to transmit producer data

<p align="center">
<image src=./firehose.JPG width=700/>
<p/>

### 3. Kinesis Video Streams

* Producers such as web cams, security cameras, audio feeds, images
* Consumers - Kinesis Video Stream applications, EC2 batch consumers
* Stores to S3

### 4. Kinesis Data Analytics

* Use SQL to process streaming data
* Sources: Kinesis Data Streams and Kinesis Data Firehose
* SQL queries output to S3, Redshift, BI tools

### 5. AWS Glue


### 6. Analyze and visualize data



:point_up_2: [back](#contents)

## Modeling

:point_up_2: [back](#contents)

## Algorithms

:point_up_2: [back](#contents)

## Implementation and Operations

:point_up_2: [back](#contents)




