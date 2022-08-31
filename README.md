# Stock Price Prediction: Apple 

## Table of Contents

[Problem Statement](#Problem-Statement) <br>
[Data Exploration](#Data-Exploration) <br>
[Data Preprocessing](#Data-Preprocessing) <br>
[Feature Engineer](#Feature-Engineer) <br>
[Machine Learning Modeling](#Machine-Learning-Modeling) 
- [Logistic Regression](#Logistic-Regression)
- [Decision Tree](#Decission-Tree)
- [Random Forest](#Random-Forest)

### Problem Statement
The task is to predict the day price direction of Apple stock, AAPL.

The stock market is very complex and highly volatile. In order to be profitable, we do not need to predict the correct price, but rather, the price direction: whether it will be higher or lower than the price that is today. If we predict it to be higher, we might as well buy some stocks, else, we should probably sell.

Therefore, the target would be a binary classification whether the next day closing price will be higher than the opening price.
### Data Exploration
Here is a slice of our data: 
![Screen Shot 2022-08-30 at 9 57 35 PM](https://user-images.githubusercontent.com/64395120/187582706-67b52f05-0013-4a6f-9cca-c50faf14ce43.png)

We have data for the period from 1997 up to year 2020 that we have split that into training (1997-2016), validation (2016-2018) and testing (2018-2020) periods. The data is available in the AMZN_train.csv, AMZN_val.csv and AMZN_test.csv files, respectively.

Each dataset has the same format with the following 7 columns:

Date - in format YYYY-MM-DD
Open - stock price upon opening of an exchange
High - the highest stock price on a given day
Low - the lowest stock price on a given day
Close - stock price at the end of a trading day
Adj Close - adjusted closing price that takes into account corporate actions
Volume - the amount of shares traded over the course of a trading day
### Data Preprocessing

### Machine Learning Modeling

Work in progress

