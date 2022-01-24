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
  * Modeling Concepts
  * Train Models
  * Evaluate and Deploy Models
  * Automatic Model Tuning
* [Algorithms](#algorithms)
  * Regression Algorithms
  * Clustering Algorithms
  * Classification Algorithms
  * Image Analysis Algorithms
  * Text Analysis Algorithms
  * Reinforcement Learning Algorithms
  * Forecasting Algorithms
* [Implementation and Operations](#implementation-and-operations)
* [Practice Questions](#practice-questions)

## Machine Learning Life Cycle

<p align="center">
<image src=./ML_cycle.png width=600/>
<p/>

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

* Fully managed service that automatically scales to match data throughput
* Capture, transform and load streaming data into S3, Redshift, Elasticsearch, and Splunk
* Automatically convert to Apache Parquet/ORC before delivering to S3
* Near real-time analytics
* Requires no ongoing administration
* Putting data into **data delivery stream**
* Transform your data using Lambda 

<p align="center">
<image src=./dsvsf.png width=500/>
<p/>

### 3. Kinesis Video Streams

* Producers such as web cams, security cameras, audio feeds, images
* Consumers - Kinesis Video Stream applications, EC2 batch consumers
* Stores to S3

### 4. Kinesis Data Analytics

* Automatically provisions and scales infrastructure to read streaming media
* Use SQL to process streaming data
* Sources: Kinesis Data Streams and Kinesis Data Firehose
* SQL queries output to S3, Redshift, BI tools

### 5. AWS Glue

**Transform and clean data**

A fully managed ETL service for categorizing, cleaning, enriching, and moving data

* Data Catalog: persistent metadata store

* Classifier: determines the schema of your data

* Connection: the properties required to connect to data store

* Crawler: connect to a data store and step through prioritized list of classifiers to determine schema

* Database: set of associated data catalog table definitions

* Data store: repository for persistently storing data

* Data source: data store used as input to transformation

* Data target: data store that a transformation writes to

**Format 1 data --> create a crawler to infer schema --> Glue generate code in python --> run python code as an ETL job --> Format 2 data**

### 6. Analyze and visualize data

* Visualize data before choosing a ML algorithm
  * Identify patterns
  * Find corrupt data
  * Identify outliers
  * Find imbalances in the data
  * Explore and demonstrate relationships

BI tootls:
1. Amazon Quicksight
2. Tensorflow with TensorBoard
3. Tableau

Quicksight import data: csv link in S3 + manifest file

:point_up_2: [back](#contents)

## Modeling

### 1. Modeling Concepts

* Supervised learning
* Unsupervised learning
* Reinforcement learning

Hyperparameter

* Model hyperparameters: influence model performance
* Algorithm hyperparameters: affect the speed and quality of learning process

### 2. Train Models

Steps:
1. Gather/engineer data into your dataset
```python
bucket_name = "your-s3-bucket-name"
s3 = boto3.resource('s3')
s3.create_bucket(Bucket=bucket_name)
```
2. Randomize the dataset
3. Split the dataset into train and test datasets
```python
train_data, test_data = np.split(model_data.sample(frac=1, random_state), [int(0.7 * len(model_data))])
```
4. Choose best algorithm
```python
role = get_execution_role()
containers = {:}
my_region = boto3.session.Session().region_name
```
5. Load container for chosen model
6. Manage compute capacity
7. Create an instance of chosen model
8. Define model's hyperparameter values
```python
sess = sagemaker.Session()
xgb = sagemaker.estimator.Estimator(role, train_instance_type, output_path)
xgb.set_hyperparameters()
```
9. Train the model
```python
xgb.fit({'train': s3_input_train})
```
### 3. Evaluate and Deploy Models

```python
xgb_predictor = xgb.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
```

SageMaker Hosting Services
1. Create model in SageMaker
2. Create an endpoint configuration
3. Create HTTPS endpoint

<p align="center">
<image src=https://docs.aws.amazon.com/sagemaker/latest/dg/images/sagemaker-architecture.png
 width=700/>
<p/>

### 4. Automatic Model Tuning

#### Hyperparameter Tuning Job

* Random search
* Bayesian search: treats hyperparameter tuning like a regression problem

#### Define Metrics

1. Built-in algorithms: choose one of the metrics as the objective metric
2. Custom algorithms: emit at least one metric to stderr or stdout

#### Define Hyperparameter Ranges

* CategoricalParameterRanges
* ContinuousParameterRanges
* IntegerParameterRanges

#### Hyperparameter Scaling

* Auto: choose the best scale
* Linear: search values in range using a linear scale
* Logarithmic
* ReverseLogarithmic

:point_up_2: [back](#contents)

## Algorithms

### 1. Regression Algorithms

* [Linear Learner](https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html)
* XGBoost
  * Gradient boosted trees
  * Predict a target by combining the estimates from a set of simpler models
  * Can differentiate the importance of features through wights
* K-Nearest Neighbors
  * Find the k closest points to the sample point and gives a prediction of the average of their features
  * Indexed based
  * Use PCA to optimize dimensionality
  * Not often used for regression, but can
 * Factorization Machines
  * Extension of linear model used on high dimensional sparse datasets
  * Used for click prediction, and item recommendation 

|Algorithms|Hyperparameter|Definition|
|---|---|---|
|Linear Learner | feature_dim | number of feature in the input |
| | predictor_type | regressor for regression problems |
| | loss | auto, squared_loss, absolute_loss, etc. |
| XGBoost | num_round | number of rounds the training runs|
| | objective | reg:logistic, reg:squarederror |
|K-Nearest Neighbors | feature_dim || 
|| k | number of nearest neighbors|
||predictor_type| regressor for regression problems |
||sample_size||
||dimensionality_reduction_target| target dimension to reduce to|
|Factorization Machines|feature_dim||
||num_factors|dimensionality of factorization|
||predictor_type||

```python
from io import StringIO
s3 = boto3.resource('s3')
bucket = 'name'
object_key = '.csv'

csv_obj = s3.Object(bucket, object_key)
csv_string = csv_obj.get()['Body'].read().decode('utf-8')

dataset = pd.read_csv(StringIO(csv_string))
dataset['dteday'] = dataset['dteday'].str.replace("-","")

train_data, test_data = np.split(dataset.sample(frac=1, random_state), [int(0.7 * len(dataset))])

feature_dataset = train_data[]
features = np.array(feature_dataset.values).astype('float32')

label_dataset = train_data[]
labels = np.array(label_dataset.values).astype('float32')
labels_vec = np.squeeze(np.asarray(labels))

# Setup protoBuf
buffer = io.BytesIO()
smac.write_numpy_to_dense_tensor(buffer, features, labels_vec)
buffer.seek(0)

boto3.resource('s3').Bucket(bucket).Object(os.path.join()).upload_fileobj(buffer)

# Model artifects
output_location

# Get the Linear Learner container instance
from sagemaker.amazon.amazon_estimator import get_image_uri
linear_container = get_image_uri(boto3.Session().region_name, 'linear-learner')

# Train the model
role = get_execution_role()
sagemaker_session = sagemaker.Session()
# Provide the container, role, instance type and model output location
linear = sagemaker.estimator.Estimator(linear_container, role, train_instance_count, train_instance_type, output_path, sagemaker_session)
linear.set_hyperparameters(future_dim, mini_batch_size, predictor_type)
linear.fit({'train': s3_training_data_location})
```

### 2. Clustering Algorithms

#### K-Means

Amazon SageMaker uses a modified version of the web-scale k-means clustering algorithm.

Objective is to minimize within-cluster sum of squares (WCSS)

<p align="center">
<image src=https://wikimedia.org/api/rest_v1/media/math/render/svg/debd28209802c22a6e6a1d74d099f728e6bd17a4 />
<p/>

### 3. Classification Algorithms

Binary-class or multiple-class

* Linear Learner
* Blazing Text: Word2vec and text classification algorithms (sentiment analysis, named entity recognition, machine translation, spam)
* XGBoost: gradient boosted trees algorithm
* K-Nearest Neighbors
* Factorization Machines: extension of linear model used on high dimensional sparse datasets (click prediction, item recommendation), scored using Binary Cross Entropy, Accuracy, F1 Score
* [Random Cut Forecast](https://medium.com/analytics-vidhya/random-cut-forest-321aae4d8a59)
  * Unsupervised algorithm for detecting anomalous data points within a data set
  * Uses an anomaly score (high score -> anomaly in the data)
  * Common practice: scores beyond 3 SD from the mean score are considered anomalous
  * Use case: find exceptions in streaming trade data

|Algorithms|Hyperparameter|Definition|
|---|---|---|
| Blazing Text | mode | Word2vec architecture used for training (batch_skipgram, skipgram, cbow)|
| Image Classification | num_classes| |
|| num_training_samples | number of training examples in the input dataset|
|| early_stopping | threshold at which stop training |
| Random Cut Forest | feature_dim ||
|| eval_metrics | score a labeled test data set (accuracy, precision_recall_fscore)
|| num_trees ||

#### AWS Random Cut Forest VS Decision Trees

* RCF randomly selects observations and features to grow trees, DT uses entirely the dataset.
* More trees, more accuracy. Uncorrelated errors will average out.
* When application requires greater accuracy and repeatability is not paramount.
* Work fine without GPU.

> Bagging: a random sample of data in a training set is selected with replacement—meaning that the individual data points can be chosen more than once.

> Ensemble Learning: after several data samples are generated, these weak models are then trained independently, and depending on the type of task—regression or classification, for example—the average or majority of those predictions yield a more accurate estimate.

> Boosting: one entity taking the value of another entity before it makes his calculations.

### 4. Image Analysis Algorithms

* Image Classification
  * ResNet (scratch or transfer learning)
  * Input format: RecordIO, .jpg, .png

* Object Detection
  * Object is categorized into one of classes, with a confidence score
  * Location and scale of the object in the image are noted by bounding box
  * Scratch or pre-trained on ImageNet dataset
 
|Algorithms|Hyperparameter|Definition|
|---|---|---|
| Object Detection| num_classes||
|| num_training_samples||
|| use_pretrained_model||


### 5. Text Analysis Algorithms

* Blazing Text
  * Can use pre-trained vector representations that improve the generalizability of ther models
* Latent Dirichlet Allocation (LDA)
  * Unsupervised learning algorithm that organizes a set of text observations into distinct categories
  * Frequently used to discover a number of topics shared across documents within a collection of texts
  * Document: each observation, Feature: count of a word in the documents
  * Topics are not specified in advance
  * Each document is described as a mixture of topics
* Neural Topic Model (NTM)
  * Unsupervised learning algorithm that organizes a corpus of documents into topics containing word groupings, based on statistical distribution of the word groupings
  * Similar to LDA, but will produce different outcomes
* Object2Vec
  * General purpose neural embedding algo that finds related clusters of words (words that are semantically similar)
  * Used for information retrieval, product search, item matching, customer profiling
  * Use case: recommendation engine based on collaborative filtering
* Seq2seq
  * Supervised learning, input of a sequence of tokens (audio, text, radar data) and output of another sequence of tokens
  * RNN and CNN 
  * SOTA encoder-decoder architecture

**Fasttext**
* Pre-trained word vectors learned on different sources
* lid.176.bin: language identification model

```python
!wget -O model.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
!tar -czvf langid.tar.gz model.bin
blazing_text_model_location = my_session.upload_data("langid.tar.gz", bucket=bucket_name, key_prefix=prefix)
!rm langid.tar.gz model.bin
```

### 6. Reinforcement Learning Algorithms

* It learns a strategy, called a policy, that optimizes an objective for an agent acting in an environment.
* Based on Markov Decision Processes models

<p align=center>
<image src=https://1.bp.blogspot.com/-DQb6CQXKyF4/YQRCqMP9mQI/AAAAAAAAkpc/bBOEN5sYh60gOoytFy39X1RXLpEICBeMQCNcBGAsYHQ/s908/rl_basics.png width=400/>
<p/>

Use cases:
* Robotics
* Traffic light control
* Predictive auto scaling
* Tuning parameters of a web system
* Optimize chemical reactions
* Personalized recommendations
* Gaming

RL in TensorFlow and Apache MXNet
* RL toolkits: SageMaker supports Intel Coach and Ray RLlib
* Environment: EnergyPlus, RoboSchool, Matlab simulink

|Hyperparameter|Definition|
|---|---|
|learning_rate| how fast the model learns|
|discount_factor| short-term or long-term rewards|
|entropy| degree of uncertainty, exploit what's already known VS exploration|

### 7. Forecasting Algorithms

* DeepAR
  * Supervised algo that forecasts 1-D time series using RNN
  * Trains a single model jointly over all of the similar time series in your dataset
  * Automatically derives time series based on the frequency of the target series

|Hyperparameter|Definition|
|---|---|
|context_length|the number of time-points that model gets to see before making the prediction|
|epochs|maximum number of passes over the training data|
|prediction_length|forecast horizon|
|time_freq|the granularity of the time series|

:point_up_2: [back](#contents)

## Implementation and Operations

* SageMaker hosting automatically scales your endpoint instances to the performance needed through Application Auto Scaling
* SageMaker automatically scaling endpoint instances scale out and spread instances across multiple availability zones
* Artifacts are encrypted in transit and at rest, request to endpoint API can be made over a secure SSL connection
* Assign IAM roles to model instance to provide permission to access resources
* Clients send HTTPS requests to endpoint to obtain inferences
* Can deploy multiple variants of a model to the same HTTPS endpoint

### AWS Services

* Lex: build chatbots, conversational interfaces using voice and text
* Transcribe: speech recognition
* Polly: text-to-speech
* Rekognition: deep learning for image and video analysis, facial recognition
* Translate: translate text
* Comprehend: insights and relationships in text, understand positive or negative sentiment of the text

### Secure SageMaker Instances

* Seure your jupyter notebook instances
  * Use EC2 instances dedicated for your use 
  * Can map SageMaker resources to VPC so you can use your network controls
  * Control access to jupyer notebooks and models through IAM
  * Can only access from within your VPC using your VPC Endpoints (private connectivity)
  * CloudWatch and CloudTrail for logging training job
 
 > Amazon Virtual Private Cloud (Amazon VPC) enables you to launch AWS resources into a virtual network that you've defined. This virtual network closely resembles a traditional network that you'd operate in your own data center, with the benefits of using the scalable infrastructure of AWS.
 
 ### Model Monitor
 
 * Monitor production model to detect deviations in data quality compared to a baseline data
   * Create baselining job
   * Create a continuous monitoring schedule
   * Start continuous monitoring

:point_up_2: [back](#contents)

## Practice Questions

### 1. Correlation

**Correlation** is a bivariate analysis that measures the strength of association between two variables and the direction of the relationship. 

*Pearson Correlation Coefficient* - Gaussian relationship between variables (normal distribution)

<p align=center>
<image src=https://wikimedia.org/api/rest_v1/media/math/render/svg/2b9c2079a3ffc1aacd36201ea0a3fb2460dc226f width=400/>
<p/>

*Spearman's Rank Correlation Coefficient* - non-Gaussian relationship between variables 

<p align=center>
<image src=https://www.statisticshowto.com/wp-content/uploads/2015/01/tied-ranks-1.png width=400/>
<p/>

*Polychoric Correlation* is used to understand the relationship of variables gathered via surveys such as personality tests and surveys that use rating scales.

### 2. Naive Bayes

In statistics, naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naïve) independence assumptions between the features (see Bayes classifier). 

<p align=center>
<image src=https://wikimedia.org/api/rest_v1/media/math/render/svg/52bd0ca5938da89d7f9bf388dc7edcbd546c118e width=300/>
<p/>

#### Gaussian Naive Bayes

When dealing with continuous data, a typical assumption is that the **continuous** values associated with each class are distributed according to a normal (or Gaussian) distribution.

#### Bernoulli Naive Bayes

Used for discrete data, where features are only in binary form. It explicitly gives penalty to the model for non-occurrence of any of the features which are necessary for predicting the output y. And the other multinomial variant of Naive Bayes ignores this features instead of penalizing.

In case of small amount of data or small documents(for example in text classification), Bernoulli Naive Bayes gives more accurate and precise results as compared to other models.

Document classification tasks where you wish to know **whether** a word from your vocabulary **appears** in your observed text or not. 

#### Multinomial Naive Bayes

With a multinomial event model, samples (feature vectors) represent the frequencies with which certain events have been generated by a multinomial (p_1, p_2, ..., p_n) where p_i is the probability that event i occurs. A feature vector x = (x_1, x_2, ..., x_n) is then a histogram, with x_i counting the number of times event i was observed in a particular instance. This is the event model typically used for document classification, with events representing the occurrence of a word in a single document (see bag of words assumption).

Document classification tasks where you wish to know the **frequency** of a given word from your vocabulary.

### 3. DeepAR

Probabilistic Forecasting with Autoregressive Recurrent Networks

The DeepAR Forecasting algo works great when you are trying to forecast using much similar time series across a set of cross-sectional units. The collective time series of other products' sales will help predict sales for the new product.

### 4. Fraud Detector Service

ONLINE_FRAUD_INSIGHTS model

### 5. CNN-QR

Amazon CNN-QR is the only forecast algorithm that accepts related time series data without future values.

Amazon Forecast CNN-QR, Convolutional Neural Network - Quantile Regression, is a proprietary machine learning algorithm for forecasting scalar (one-dimensional) time series using causal convolutional neural networks (CNNs). This supervised learning algorithm trains one global model from a large collection of time series and uses a quantile decoder to make probabilistic predictions.

### 6. Lasso

Lasso regularization (L1) handles outliers well, better than Ridge (L2). Lasso combats overfitting by shrinking the parameters towards 0. This makes some features obsolete. 

Ridge (in regression problems), combats overfitting by forcing weights to be small, but not making them exactly 0. A major snag to consider when using L2 regularization is that it’s not robust to outliers. The squared terms will blow up the differences in the error of the outliers. The regularization would then attempt to fix this by penalizing the weights. 

<p align=center>
<image src=https://miro.medium.com/max/2000/1*zMLv7EHYtjfr94JOBzjqTA.png width=500/>
<p/>

#### Key Differences

* L1 regularization penalizes the sum of absolute values of the weights, whereas L2 regularization penalizes the sum of squares of the weights. 
* The L1 regularization solution is sparse. The L2 regularization solution is non-sparse.
* L2 regularization doesn’t perform feature selection, since weights are only reduced to values near 0 instead of 0. L1 regularization has built-in feature selection.
* L1 regularization is robust to outliers, L2 regularization is not. 

### 7. Clustering Metrics

#### rand_core

The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings. 1.0 stands for perfect match.

The raw RI score is:

RI = (number of agreeing pairs) / (number of pairs)

#### adjusted_mutual_info_score

Mutual Information is a function that measures the agreement of the two assignments, ignoring permutations. Perfect labeling is scored 1.0

#### completeness_score

Completeness metric of a cluster labeling given a ground truth.

A clustering result satisfies completeness if all the data points that are members of a given class are elements of the same cluster.

### 8. 

:point_up_2: [back](#contents)


