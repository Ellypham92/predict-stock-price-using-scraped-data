# Stock Price Prediction: Apple 
<div id="header" align="center">
<img width="350" alt="image" src="https://user-images.githubusercontent.com/64395120/187584766-4d9f469c-9974-40b3-be08-f37404561f25.png">
</div>

## Table of Contents
[Problem Statement](#Problem-Statement) <br>
[Data Exploration](#Data-Exploration) <br>
[Data Preprocessing](#Data-Preprocessing) <br>
[Feature Engineer](#Feature-Engineer) <br>
[Machine Learning Modeling](#Machine-Learning-Modeling) 
- [Logistic Regression](#Logistic-Regression)
- [Decision Tree](#Decission-Tree)
- [Random Forest](#Random-Forest)

### :iphone: Problem Statement
The task is to predict the day price direction of Apple stock, AAPL.

The stock market is very complex and highly volatile. In order to be profitable, we do not need to predict the correct price, but rather, the price direction: whether it will be higher or lower than the price that is today. If we predict it to be higher, we might as well buy some stocks, else, we should probably sell.

Therefore, the target would be a binary classification whether the next day closing price will be higher than the opening price.
### :tada: Data Exploration
Here is a slice of our data using describe() method: 
![Screen Shot 2022-08-30 at 9 57 35 PM](https://user-images.githubusercontent.com/64395120/187582706-67b52f05-0013-4a6f-9cca-c50faf14ce43.png)

### :baseball: Data Preprocessing <br>

 ##### We will drop the column Unamed ) using drop() method since it is not helpful and split data into
- Training (1980-12-12 to 2005-12-12)
- Validation (2005-12-13 to 2013-12-13)
- Testing (2013-12-14 to 2022-08-29)

Sample of training data: 
![Screen Shot 2022-08-30 at 10 03 38 PM](https://user-images.githubusercontent.com/64395120/187583518-a315c5a5-069d-475b-8b64-48b848dafcff.png)

Each dataset has the same format with the following 7 columns:

Date - in format YYYY-MM-DD <br>
Open - stock price upon opening of an exchange <br>
High - the highest stock price on a given day <br>
Low - the lowest stock price on a given day <br>
Close - stock price at the end of a trading day <br>
Volume - the amount of shares traded over the course of a trading day <br>
Dividends -  <br>
Stock Splits - <br>

##### We change the date type for date to datetime and Volume to float using astype() method:
![Screen Shot 2022-08-30 at 10 08 09 PM](https://user-images.githubusercontent.com/64395120/187584134-e2d9dd14-11f2-4321-8e01-c8596728c8ee.png)

##### We created a custom function make_graph() to plot train, validate, test data for high price/low price and volume 
![Screen Shot 2022-08-30 at 10 16 32 PM](https://user-images.githubusercontent.com/64395120/187585162-76a20fba-f766-469a-9f80-459cb58dacad.png)


### :bicyclist: Machine Learning Modeling

Work in progress

