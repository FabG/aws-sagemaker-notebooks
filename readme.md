# AWS SageMaker Notebooks

This repo includes several notebooks from SageMaker to showcase different algorithms


### 1. Intro to SageMaker algorithms

### 1.1 Time series Forecasting (DeepAR)
#### Overview
The Amazon SageMaker DeepAR forecasting algorithm is a ***supervised*** learning algorithm for **forecasting scalar (one-dimensional) time series** using ***RNN*** (recurrent neural networks.

Classical forecasting methods, such as autoregressive integrated moving average (ARIMA) or exponential smoothing (ETS), fit a single model to each individual time series. They then use that model to extrapolate the time series into the future.

More info at:
- [SageMaker Deep AR documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html)
- [Deep AR - how it works](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar_how-it-works.html)


#### Notebooks
 - [DeepAR on Synthetic timeseries](1-intro-sagemaker-algos/forecasting/deepar_synthetic.ipynb) - forecasting on univariate synthetic time series using DeepAR and publishing/leveraging a SM endpoint to run predictions



### 1.2 Multi-Class Classifier (XGBoost)
#### Overview
The Amazon SageMaker **XGboost** (eXtreme Gradient Boosting) algorithm is a popular and efficient open-source implementation of the **gradient boosted trees** algorithm.

Gradient boosting is a **supervised learning** algorithm that attempts to accurately predict a target variable by combining an **ensemble of estimates from a set of simpler and weaker models**.

The XGBoost algorithm performs well in machine learning competitions because of its robust handling of a variety of data types, relationships, distributions, and the variety of hyperparameters that you can fine-tune. You can use XGBoost for **regression, classification (binary and multiclass), and ranking problems**.


More info at: [SageMaker XGboost documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html)



#### Notebooks
- [XGboost Multi-Classification on MNIST dataset](1-intro-sagemaker-algos/classification/xgboost_mnist.ipynb)


### 1.3 Regression (XGBoost)
#### Overview
The Amazon SageMaker **XGboost** (eXtreme Gradient Boosting) algorithm is a popular and efficient open-source implementation of the **gradient boosted trees** algorithm.

Gradient boosting is a **supervised learning** algorithm that attempts to accurately predict a target variable by combining an **ensemble of estimates from a set of simpler and weaker models**.

The XGBoost algorithm performs well in machine learning competitions because of its robust handling of a variety of data types, relationships, distributions, and the variety of hyperparameters that you can fine-tune. You can use XGBoost for **regression, classification (binary and multiclass), and ranking problems**.


More info at: [SageMaker XGboost documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html)



#### Notebooks
- [XGboost Regression on Abalone dataset](1-intro-sagemaker-algos/regression/xgboost_regression_abalone.ipynb)


### 1.4.a NLP - Topics modeling (LDA)
#### Overview
The Amazon Sagemaker **LDA** (Latent Dirichlet Allocation) algorithm is an **unsupervised learning** algorithm that attempts to describe a set of observations as a mixture of distinct **categories**.

LDA is most commonly used to perform "topic modeling" and discover a user-specified number of **topics shared by documents within a text corpus**. Here each observation is a document, the features are the presence (or occurrence count) of each word, and the categories are the topics. Since the method is unsupervised, the **topics are not specified up front, and are not guaranteed to align with how a human may naturally categorize documents**. The topics are learned as a probability distribution over the words that occur in each document. Each document, in turn, is described as a mixture of topics.


LDA supports both **recordIO-wrapped-protobuf** (dense and sparse) and **CSV** file formats. LDA currently only supports single-instance CPU training. CPU instances are recommended for hosting/inference.

More info at: [SageMaker LDA documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/lda.html)

#### Notebooks
- [LDA Topic modeling on synthetic dataset](1-intro-sagemaker-algos/nlp/lda-topic-modeling-synthetic.ipynb) - this notebook is still WIS as the synthetic data generation is failing.


### 1.4.b NLP - Topics modeling (NTM)
#### Overview
The Amazon Sagemaker **NTM** (Neural Topic Model) is an **unsupervised learning** algorithm that is used to organize a corpus of documents into *topics* that contain word groupings based on their statistical distribution.

Documents that contain frequent occurrences of words such as "bike", "car", "train", "mileage", and "speed" are likely to share a topic on "transportation" for example. Topic modeling can be used to classify or summarize documents based on the topics detected or to retrieve information or recommend content based on topic similarities. The topics from documents that NTM learns are characterized as a latent representation because the topics are inferred from the observed word distributions in the corpus. The semantics of topics are usually inferred by examining the top ranking words they contain. Because the method is unsupervised, only the number of topics, not the topics themselves, are prespecified. In addition, the topics are not guaranteed to align with how a human might naturally categorize documents.


More info at: [SageMaker NTM documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/ntm.html)

#### Notebooks
- [NTM Topic modeling on synthetic dataset](1-intro-sagemaker-algos/nlp/ntm-topic-modeling-synthetic.ipynb) - this notebook is still WIS as the synthetic data generation is failing.


### 2. Resources
- [amazon-sagemaker-examples](https://github.com/awslabs/amazon-sagemaker-examples)
