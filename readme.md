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



### 1.5a Classification/Regression on high dimension sparse dataset (Factorization Machines)
#### Overview
A **factorization machine** is a general-purpose **supervised learning** algorithm that you can use for both classification and regression tasks.

It is an extension of a linear model that is designed to capture interactions between features within high dimensional sparse datasets economically. For example, in a click prediction system, the factorization machine model can capture click rate patterns observed when ads from a certain ad-category are placed on pages from a certain page-category.

Factorization machines are a good choice for tasks dealing with **high dimensional sparse datasets**, such as **click prediction and item recommendation**.


More info at: [SageMaker Factorization Machines documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/fact-machines.html)


#### Notebooks
 - [Factorization Machine on MNIST](1-intro-sagemaker-algos/classification_regression/factorization-machines-mnist.ipynb)


### 1.5b Classification/Regression (K-means)
#### Overview
Amazon SageMaker **k-nearest neighbors (k-NN)** algorithm is an **index-based** algorithm. It uses a non-parametric method for **classification or regression**.

- For **classification** problems, the algorithm queries the k points that are closest to the sample point and returns the most frequently used label of their class as the predicted label.
- For **regression** problems, the algorithm queries the k closest points to the sample point and returns the average of their feature values as the predicted value.

Training with the k-NN algorithm has three steps: sampling, dimension reduction, and index building. The main objective of k-NN's training is to construct the index. The index enables efficient lookups of distances between points whose values or class labels have not yet been determined and the k nearest points to use for inference.

More info at: [SageMaker K-NN documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/k-nearest-neighbors.html)


#### Notebooks
 - [K-NN)](1-intro-sagemaker-algos/classification_regression/k_nearest_neighbors_covtype.ipynb) - Notebook showcasing Multi-Class Classification, with a goal to predict the forest coverage type given a location



### 1.5c Classification/Regression (Linear Learner)
#### Overview
Amazon SageMaker **Linear Learner** is an **supervised learning** algorithm. For input, you give the model labeled examples (x, y). x is a high-dimensional vector and y is a numeric label.
 - For binary classification problems, the label must be either 0 or 1.
 - For multiclass classification problems, the labels must be from 0 to `num_classes - 1`.
 - For regression problems, y is a real number.

 The algorithm learns a linear function, or, for classification problems, a linear threshold function, and maps a vector x to an approximation of the label y.

The Amazon SageMaker linear learner algorithm provides a solution for both **classification and regression** problems. With the SageMaker algorithm, you can simultaneously explore different training objectives and choose the best solution from a validation set. You can also explore a large number of models and choose the best. The best model optimizes either of the following:
- Continuous objectives, such as mean square error, cross entropy loss, absolute error.
- Discrete objectives suited for classification, such as F1 measure, precision, recall, or accuracy.

Compared with methods that provide a solution for only continuous objectives, the SageMaker linear learner algorithm provides a significant increase in speed over naive hyperparameter optimization techniques. It is also more convenient.
Amazon SageMaker's Linear Learner algorithm extends upon typical linear models by training many models in parallel, in a computationally efficient manner. Each model has a different set of hyperparameters, and then the algorithm finds the set that optimizes a specific criteria. This can provide substantially more accurate models than typical linear algorithms at the same, or lower, cost.


More info at: [SageMaker Linear Learner documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html)


#### Notebooks
  - [Linear Learner with MNIST](1-intro-sagemaker-algos/classification_regression/linear-learner-mnist.ipynb) - This notebook will leverage Linear Learner to perform a binary classification (if a hand digit number is a 0)



### 1.6a Whole Image Multi-Label classification (Image Classification)
#### Overview
The Amazon SageMaker **Image Classification** algorithm is a **supervised learning** algorithm that supports **multi-label classification**.

It takes an image as input and outputs one or more labels assigned to that image. It uses a **convolutional neural network (ResNet)** that can be:
 - trained from scratch
 - or trained using transfer learning when a large number of training images are not available.
The recommended input format for the Amazon SageMaker image classification algorithms is Apache MXNet `RecordIO`. However, you can also use raw images in `.jpg` or `.png` format.


More info at: [SageMaker Image Classification documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html)


#### Notebooks
  - [Image Classification on caltech-256](1-intro-sagemaker-algos/image_classification/image-classification-fulltraining-highlevel.ipynb) - This notebook leverages [SageMaker Neo](https://aws.amazon.com/sagemaker/neo/) to run twice as fast and reduce memory footprint, but has a compilation error. The training job finished though so it's more an error with the Neo compilation.

 - [Image Classification on caltech-256 (2)](1-intro-sagemaker-algos/images/image_classification/image_classification/Image-classification-fulltraining.ipynb) - This notebook trains MXNet on caltech256 images and expose an endpoint for real tiem inferences


### 1.6b Object recognition in Images (Object Detection)
#### Overview
The Amazon SageMaker **Object Detection** algorithm **detects and classifies objects in images** using a single deep neural network. It is a **supervised learning** algorithm that takes images as input and identifies all instances of objects within the image scene.

The object is categorized into one of the classes in a specified collection with a confidence score that it belongs to the class. Its location and scale in the image are indicated by a rectangular bounding box. It uses the Single Shot multibox Detector (SSD) framework and supports two base networks: VGG and ResNet. The network can be trained from scratch, or trained with models that have been pre-trained on the ImageNet dataset.

More info at: [SageMaker Object Detection documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/object-detection.html)

#### Notebooks
 - [Object Detection on Birds](1-intro-sagemaker-algos/object_detection/images/object_detection/object_detection_birds.ipynb)


### 1.6c Image's Pixels classification (Semantic Segmentation)
#### Overview
The SageMaker **semantic segmentation** algorithm provides a fine-grained, pixel-level approach to developing computer vision applications. It **tags every pixel in an image with a class label from a predefined set of classes**.
The output is an integer matrix (or a grayscale image) with the same shape as the input image. This output image is also called a segmentation mask.

Tagging is fundamental for understanding scenes, which is critical to an increasing number of computer vision applications, such as self-driving vehicles, medical imaging diagnostics, and robot sensing.

For comparison:
 - the SageMaker *Image Classification* Algorithm is a supervised learning algorithm that analyzes only whole images, classifying them into one of multiple output categories.
 - the *Object Detection* Algorithm is a supervised learning algorithm that detects and classifies all instances of an object in an image. It indicates the location and scale of each object in the image with a rectangular bounding box.

Because the semantic segmentation algorithm classifies every pixel in an image, it also provides information about the shapes of the objects contained in the image. The segmentation output is represented as a grayscale image, called a segmentation mask. A segmentation mask is a grayscale image with the same shape as the input image.

More info at: [SageMaker Semantic Segmentation documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/semantic-segmentation.html)

#### Notebooks
 - [Object Detection on Birds](1-intro-sagemaker-algos/semantic-segmentation/images/semantic_segmentation/semantic-segmentation-pascalvoc.ipynb) - training a fully-convolutional network (FCN) on the Pascal VOC dataset. Need to try it again as the Pasval VOC website is down for downloading the training data...


### 1.7a Anomaly Detection (Random Cut Forest)
#### Overview
The Amazon SageMaker **Random Cut Forest (RCF)**  is an **unsupervised algorithm** for **detecting anomalous data points** within a data set. These are observations which diverge from otherwise well-structured or patterned data.

Anomalies can manifest as unexpected spikes in time series data, breaks in periodicity, or unclassifiable data points. They are easy to describe in that, when viewed in a plot, they are often easily distinguishable from the "regular" data. Including these anomalies in a data set can drastically increase the complexity of a machine learning task since the "regular" data can often be described with a simple model.

With each data point, RCF associates an `anomaly score`.
- **Low score values** indicate that the data point is considered **"normal."**
- **High values** indicate the presence of an **anomaly** in the data. The definitions of "low" and "high" depend on the application but **common practice suggests that scores beyond three standard deviations from the mean score are considered anomalous**.

More info at: [SageMaker Random Cut Forest documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html)


#### Notebooks
 - [Anomaly detection (RCF) on NYC Taxi dataset](1-intro-sagemaker-algos/random_cut_forest/random_cut_forest.ipynb)


### 1.7b Anomaly Detection on IP Addresses (IP Insights)
#### Overview

Amazon SageMaker **IP Insights** is an **unsupervised learning** algorithm that learns the **usage patterns for IPv4 addresses**.

It is designed to capture associations between IPv4 addresses and various entities, such as user IDs, hostnames or account numbers. You can use it to identify a user attempting to log into a web service from an anomalous IP address, for example. Or you can use it to identify an account that is attempting to create computing resources from an unusual IP address.

Under the hood, it learns **vector representations for online resources and IP addresses**. This essentially means that if the vector representing an IP address and an online resource are close together, then it is likely for that IP address to access that online resource, even if it has never accessed it before.



More info at: [SageMaker IP Insights documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/ip-insights.html)


#### Notebooks
  - [IP Insights on Apache Web Logs](1-intro-sagemaker-algos/ip_insights/ipinsights-weblogs.ipynb)
  This notebook, in addition to training and hosting the mode as an endpoint, also includes the:
    - [Amazon Automatic Model Tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html) to pick the appropriate hyperparameters.
    - [Amazon Batch Transform](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-batch.html) to get inferences on an entire dataset in S3


### 1.8 Clustering (K-Means)
#### Overview

Amazon SageMaker **K-Means** is an **unsupervised learning** algorithm that can be used for clustering. It attempts to find **discrete groupings** within data, where members of a group are as similar as possible to one another and as different as possible from members of other groups. You define the attributes that you want the algorithm to use to determine similarity.

Amazon SageMaker uses a modified version of the web-scale k-means clustering algorithm. Compared with the original version of the algorithm, the version used by Amazon SageMaker is more accurate. Like the original algorithm, it scales to massive datasets and delivers improvements in training time. To do this, the version used by Amazon SageMaker streams mini-batches (small, random subsets) of the training data.


More info at: [SageMaker K-Means documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/k-means.html)


#### Notebooks
  - [K-Means on Census data](1-intro-sagemaker-algos/clustering/k-means-countycensus.ipynb) - This notebooks leverages PCA before using K-Means to



### 1.9 Embeddings (Object2Vec)
#### Overview

Amazon SageMaker **Object2Vec** is is a general-purpose **neural embedding** algorithm that is highly customizable. It can learn low-dimensional dense embeddings of high-dimensional objects.

The embeddings are learned in a way that preserves the semantics of the relationship between pairs of objects in the original space in the embedding space. You can use the learned embeddings to efficiently compute nearest neighbors of objects and to visualize **natural clusters of related objects in low-dimensional space**, for example. You can also use the embeddings as features of the corresponding objects in downstream supervised tasks, such as classification or regression.

Object2Vec generalizes the well-known Word2Vec embedding technique for words that is optimized in the SageMaker BlazingText algorithm.

More info at: [SageMaker Object2Vec documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/object2vec.html)


#### Notebooks
  - [Object2Vec on Movie](1-intro-sagemaker-algos/object2vec/object2vec-movie-recommendation-movielens100k.ipynb) - This notebooks learns embeddings from User-Movie pairs


### 1.10 Object Tokenization (Sequence-to-Sequence)
#### Overview

Amazon SageMaker **Sequence to Sequence** is a **supervised learning** algorithm where:
- the **input** is a **sequence of tokens** (for example, text, audio)
- the **output** generated is **another sequence of tokens**.

Example applications include:
 - *machine translation* (input a sentence from one language and predict what that sentence would be in another language)
 - *text summarization* (input a longer string of words and predict a shorter string of words that is a summary)
 - *speech-to-text* (audio clips converted into output sentences in tokens).

Recently, problems in this domain have been successfully modeled with deep neural networks that show a significant performance boost over previous methodologies. Amazon SageMaker **seq2seq** uses **Recurrent Neural Networks (RNNs) and Convolutional Neural Network (CNN)** models with attention as encoder-decoder architectures.


More info at: [SageMaker Seq2seq documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/seq-2-seq.html)


#### Notebooks
 - [Seq2Seq for English-German translation](1-intro-sagemaker-algos/seq2seq/seq2seq-translation-english-german.ipynb)


### 1.11 Dimension reduction (PCA)
#### Overview
**PCA** is an unsupervised machine learning algorithm that attempts to **reduce the dimensionality (number of features)** within a dataset while still retaining as much information as possible.
This is done by finding a *new set of features called components*, which are composites of the original features that are *uncorrelated with one another*. They are also constrained so that the *first component accounts for the largest possible variability* in the data, the second component the second most variability, and so on.

PCA is most commonly used as a pre-processing step. Statistically, many models assume data to be low-dimensional. In those cases, the output of PCA will actually include much less of the noise and subsequent models can be more accurate.

In Amazon SageMaker, PCA operates in two modes, depending on the scenario:
- **regular**: For datasets with *sparse* data and a moderate number of observations and features.
- **randomized**: For datasets with both a *large number* of observations and features. This mode uses an approximation algorithm.

PCA uses *tabular* data.


More info at: [SageMaker PCA documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/pca.html)


#### Notebooks
- [PCA](1-intro-sagemaker-algos/dimension_reduction/seq2seq/pca-mnist.ipynb)



### 2. Intro to SageMaker Applied Machine learning

#### 2.1 Breast Cancer Prediction (Linear Learner)

This [cancer prediction notebook](2-intro-sagemaker-applied-ml/breast-cancer-prediction.ipynb) illustrates how one can use SageMaker's algorithms for solving applications which require **linear models** for prediction. For this illustration, we have taken an example for breast cancer prediction using [UCI'S breast cancer diagnostic data set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).
The purpose here is to use this data set to build a predictive model of whether a breast mass image indicates benign or malignant tumor.

As a reminder, Amazon SageMaker's **Linear Learner actually fits many models in parallel**, each with slightly different hyperparameters, and then returns the one with the best fit. This functionality is automatically enabled.

#### 2.2 Traffic Violation Predictions (DeepAR)
This [traffic violation prediction notebook](2-intro-sagemaker-applied-ml/deepar-chicago-traffic-violations.ipynb) demonstrates **time series forecasting** using the Amazon SageMaker **DeepAR** algorithm by analyzing city of [Chicago’s Speed Camera Violation dataset](https://data.cityofchicago.org/Transportation/Red-Light-Camera-Violations/spqx-js37). 

### 3. Resources
- [amazon-sagemaker-examples](https://github.com/awslabs/amazon-sagemaker-examples)
