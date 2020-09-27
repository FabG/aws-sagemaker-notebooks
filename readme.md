# AWS SageMaker Notebooks

This repo includes several notebooks from SageMaker to showcase different algorithms


### 1. Intro to SageMaker algorithms

# Time series Forecasting (DeepAR)
### Overview
The Amazon SageMaker DeepAR forecasting algorithm is a ***supervised*** learning algorithm for **forecasting scalar (one-dimensional) time series** using ***RNN*** (recurrent neural networks.

Classical forecasting methods, such as autoregressive integrated moving average (ARIMA) or exponential smoothing (ETS), fit a single model to each individual time series. They then use that model to extrapolate the time series into the future.

More info at:
- [SageMaker Deep AR documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html)
- [Deep AR - how it works](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar_how-it-works.html)



### Notebooks
 - [deepar_synthetic](1-intro-sagemaker-algos/forecasting/deepar_synthetic.ipynb) - forecasting on univariate synthetic time series using DeepAR and publishing/leveraging a SM endpoint to run predictions



 ### Resources
 - [amazon-sagemaker-examples](https://github.com/awslabs/amazon-sagemaker-examples)
