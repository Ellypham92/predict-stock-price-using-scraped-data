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
Here is a slice of our data using describe() method: 
![Screen Shot 2022-08-30 at 9 57 35 PM](https://user-images.githubusercontent.com/64395120/187582706-67b52f05-0013-4a6f-9cca-c50faf14ce43.png)

We will drop the column Unamed ) using drop() method since it is not helpful and split data into:
- Training (1980-12-12 to 2005-12-12)
- Validation (2005-12-13 to 2013-12-13)
- Testing (2013-12-14 to 2022-08-29)

Sample of training data: 
![Screen Shot 2022-08-30 at 9 57 35 PM](https://user-images.githubusercontent.com/64395120/187583225-bec52d91-ddea-499a-85b8-a06b4b22f348.png)

Each dataset has the same format with the following 7 columns:

Date - in format YYYY-MM-DD
Open - stock price upon opening of an exchange
High - the highest stock price on a given day
Low - the lowest stock price on a given day
Close - stock price at the end of a trading day
Volume - the amount of shares traded over the course of a trading day
Dividends - 
Stock Splits - 
### Data Preprocessing

### Machine Learning Modeling

Work in progress

