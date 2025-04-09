# predicting-stock-prices
Dataset: https://www.kaggle.com/datasets/guillemservera/aapl-stock-data

Attempt at making a machine learning model to predict stock price trends.

## Research Question:
To predict future price movements of AAPL stock and try to make an effective model to show 
that the stock market moves following patterns.

# Predicting AAPL Stock Trends
The objective here is to predict the prices of AAPL Stock using historical data. Although our dataset has many features, we only use the closing price feature for our model. This is because of limitations with plotting the predictions. We also compute and plot the 100 day and 200 day moving averages to show the trends of the dataset.

![image](https://github.com/user-attachments/assets/9f628944-99ba-42cc-aae6-ed44660919ae)

Here, we scale the data after splitting. This is because if we scale the data before we create a bias in the predictions as the train part of the data would be scaled alongside the test part. After splitting we scale only the training dataset and then split this into a X and Y dataframes. The X train has all the datapoints from the training dataframe minus 100 datapoints from the end; these 100 datapoints are stored withing the X train in a new column making it a 3D array. And the Y train has all the datapoints all in a single column.

We then create an Artificial Neural Network model with 4 hidden layers using RELU activation on all of them. We also include dropouts on all the hidden layers with values starting from 0.2 increasing in increments of 0.1. We are going to calculate the loss with MSE for this model and train it with an epoch size of 50.

We then scale the testing dataframe and split it among X test and Y test, where X test has all the datapoints and the 100 datapoints from the training dataframe in a new column making it a 3D array, and the Y test having all the datapoints from the testing dataframe.

We then run our model to make our predictions and evaluate the results:

![image](https://github.com/user-attachments/assets/0cbe471e-2620-41c0-928f-770cd06c3807)

# Conclusion and Analysis
Here we can see that the predicted values are very similar to the real values and is giving us promising results. However, we also have to understand that there is a statistical bias formed in the model. This is due to the usage of ‘MinMaxScaler’.

## What does ‘MinMaxScaler’ do? 
While scaling the dataframe, what this does is it takes the highest value in the dataframe and equates it to a value of 1 and the lowest value to 0, from this is equates everything in the middle to in between 0 and 1 accordingly.

## Why is this incorrect?
This is incorrect because our train dataset may not have the highest value which means our model would be able to assume that the price goes higher in the test dataset; creating a bias.

## What have we done to minimise the bias? 
We have split and scaled the train and test datasets separately to minimise the bias. Although, even after doing so we can see some form of data leakage in the model and hence it cannot be called an accurate model to predict stock price trends.

