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

### 1.4.c NLP - Text classification modeling (BlazingText)
#### Overview
The Amazon SageMaker **BlazingText** algorithm provides highly optimized implementations of the **Word2vec and text classification** algorithms.

Word2vec generally is an **unsupervised learning** algorithm, designed by Google developers and released in 2013, to learn vector representations of words. The main idea is to encode words with close meaning that can substitute each other in a context as close vectors in an X-dimensional space.

The Word2vec algorithm is **useful for many downstream natural language processing (NLP) tasks, such as sentiment analysis, named entity recognition, machine translation, etc.** Text classification is an important task for applications that perform web searches, information retrieval, ranking, and document classification.

The Word2vec algorithm **maps words to high-quality distributed vectors**. The resulting vector representation of a word is called a **word embedding**. Words that are semantically similar correspond to vectors that are close together. That way, word embeddings capture the semantic relationships between words.

How Word2Vec works?
The algorithm tries to reflect the meaning of a word by analyzing its context. The algorithm exists in two flavors: CBOW and Skip-Gram. In the second approach, looping over a corpus of sentences, model tries to use the current word to predict its neighbors, or in CBOW it tries to predict the current word with the help of each of the contexts.

With the **BlazingText** algorithm, you can scale to large datasets easily. Similar to Word2vec, it provides the Skip-gram and continuous bag-of-words (CBOW) training architectures. BlazingText's implementation of the supervised multi-class, multi-label text classification algorithm extends the fastText text classifier to **use GPU acceleration with custom CUDA kernels**. You can train a model on more than a billion words in a couple of minutes using a multi-core CPU or a GPU.


More info at: [SageMaker BlazingText documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext.html)


#### Notebooks
 - [BlazingText Topic Extraction on dbpedia](1-intro-sagemaker-algos/nlp/blazingtext-classification-dbpedia.ipynb)
 - [BlazingText Pretrained Language Identification](1-intro-sagemaker-algos/nlp/blazingtext-pretrained-language-identification.ipynb)
 - [Blazing Test subwords on Text8](1-intro-sagemaker-algos/nlp/blazingtext-word2vec-subwords-text8.ipynb)



### 1.5 Classification/Regression on high dimension sparse dataset (Factorization Machines)
#### Overview
A **factorization machine** is a general-purpose **supervised learning** algorithm that you can use for both classification and regression tasks.

It is an extension of a linear model that is designed to capture interactions between features within high dimensional sparse datasets economically. For example, in a click prediction system, the factorization machine model can capture click rate patterns observed when ads from a certain ad-category are placed on pages from a certain page-category.

Factorization machines are a good choice for tasks dealing with **high dimensional sparse datasets**, such as **click prediction and item recommendation**.



More info at: [SageMaker Factorization Machines documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/fact-machines.html)


#### Notebooks
 - [Factorization Machine on MNIST](1-intro-sagemaker-algos/high_dimensional_sparse_datasets/factorization-machines-mnist.ipynb)


### 1.6 Image Multi-Label classification (Image Classification)
#### Overview
The Amazon SageMaker **Image Classification** algorithm is a **supervised learning** algorithm that supports **multi-label classification**.

It takes an image as input and outputs one or more labels assigned to that image. It uses a **convolutional neural network (ResNet)** that can be:
 - trained from scratch
 - or trained using transfer learning when a large number of training images are not available.
The recommended input format for the Amazon SageMaker image classification algorithms is Apache MXNet `RecordIO`. However, you can also use raw images in `.jpg` or `.png` format.


More info at: [SageMaker Image Classification documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html)


#### Notebooks
  - [Image Classification on caltech-256](1-intro-sagemaker-algos/image_classification/image-classification-fulltraining-highlevel.ipynb) - still WIS as I did not get the ml.p2.xlarge instance enabled for training


### 1.7 Anomaly Detection on IP Addresses (IP Insights)
#### Overview

Amazon SageMaker **IP Insights** is an **unsupervised learning** algorithm that learns the **usage patterns for IPv4 addresses**.

It is designed to capture associations between IPv4 addresses and various entities, such as user IDs, hostnames or account numbers. You can use it to identify a user attempting to log into a web service from an anomalous IP address, for example. Or you can use it to identify an account that is attempting to create computing resources from an unusual IP address.

Under the hood, it learns **vector representations for online resources and IP addresses**. This essentially means that if the vector representing an IP address and an online resource are close together, then it is likely for that IP address to access that online resource, even if it has never accessed it before.



More info at: [SageMaker IP Insights documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/ip-insights.html)


#### Notebooks
  - [IP Insights on Apache Web Logs](1-intro-sagemaker-algos/ip_insights/ipinsights-weblogs.ipynb) - still WIS as I did not get the ml.p3.2xlarge instance enabled for training


### 2. Resources
- [amazon-sagemaker-examples](https://github.com/awslabs/amazon-sagemaker-examples)
