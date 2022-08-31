# Stock Price Prediction: Apple 

Elly Pham
Created on: August 31, 2022 

## Table of Contents
[Problem Statement](#Problem-Statement) <br>
[Data Exploration](#Data-Exploration) <br>
[Data Preprocessing](#Data-Preprocessing) <br>
[Feature Engineering](#Feature-Engineer) <br>
[Machine Learning Modeling](#Machine-Learning-Modeling) 
- [Logistic Regression](#Logistic-Regression)
- [Decision Tree](#Decission-Tree)
- [Random Forest](#Random-Forest) <br>
- [Gradient Boosting Ensemble](#Gradient-Boosting-Ensemble)

[Deep Learning](#Deep-Learning) <br>
[Conclusion](#Conclusion)

###  Problem Statement
The task is to predict the day price direction of Apple stock, AAPL.

The stock market is very complex and highly volatile. In order to be profitable, we do not need to predict the correct price, but rather, the price direction: whether it will be higher or lower than the price that is today. If we predict it to be higher, we might as well buy some stocks, else, we should probably sell.

Therefore, the target would be a binary classification whether the next day closing price will be higher than the opening price.
### :tada: Data Exploration
Here is a slice of our data using describe() method: 
![Screen Shot 2022-08-30 at 9 57 35 PM](https://user-images.githubusercontent.com/64395120/187582706-67b52f05-0013-4a6f-9cca-c50faf14ce43.png)

###  Data Preprocessing <br>

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
![Screen Shot 2022-08-30 at 10 25 09 PM](https://user-images.githubusercontent.com/64395120/187585950-c2ee3c31-19f2-4e75-bed9-0d2013bee5b7.png)

![Screen Shot 2022-08-30 at 10 16 32 PM](https://user-images.githubusercontent.com/64395120/187585162-76a20fba-f766-469a-9f80-459cb58dacad.png)


### Feature Engineering  

Since we wanted to predict the next day and find out whether the next day's close is higher than the next day's open, what we need to do next is to compare the closing and the opening prices one day in advance. <br>
<br>
To do that we are going to move the day to one day before that, meaning we shift the data of the next day to one day back. Additionally, we will add a classification column - `Target`. If the closing price is greater than opening price will be 1, otherwise it will be 0. <br>
<br>
We are using shift() method here.
![Screen Shot 2022-08-31 at 1 43 50 AM](https://user-images.githubusercontent.com/64395120/187611259-6481a0ae-4ab0-40bf-8be4-c2cf65dabb4c.png)

Now, we are using value_counts() method to sum the total number of days that has higher closing pricer:
![Screen Shot 2022-08-31 at 1 46 41 AM](https://user-images.githubusercontent.com/64395120/187611829-8a8e0136-ab46-41f3-9167-a8501f31de08.png)

Since the stock price data is time series data, the price in the next day depends on price from the previous day. We are going to calculate 3 and 7 day moving average. We can use rolling(). <br>
<br>
We will add new features: 3 and 7 days moving average in the dataset that are computed rolling mean with a window length of 3 and 7 observations.
<br>
Additionally, we will add a price direction feature which is the differenece between closing price and the opening price to try if that would help for our prediction. And, price range which is the difference between high price and low price.
<br>

![Screen Shot 2022-08-31 at 1 49 55 AM](https://user-images.githubusercontent.com/64395120/187612239-8eb2f7c0-729b-41ff-aefb-c9490191749c.png)

### Machine Learning Modeling
#### Logistic Regression
Now, the Y variable is classified into 2 values 0/1 so we can test out with **logistic regression** algorithm first. <br> 
<br>
ROC is commonly used to examine the trade-off between the detection of true positives, while avoiding false positives. ROC curve plots TPR on the y-axis against FPR on the x-axis
- FPT = FP / N
- TPR = TP / P
<br>
We will compare model by using the AUV score. AUC uses a system similar to academic letter grades:
-  A: Outstanding = 0.9 to 1.0 <br>
-  B: Excellent/ good = 0.8 to 0.9 <br>
-  C: Acceptable/ fair = 0.7 to 0.8 <br>
-  D:Poor=0.6 to 0.7 <br>
-  E: No discrimination = 0.5 to 0.6 <br>
<br>

<img width="700" alt="image" src="https://user-images.githubusercontent.com/64395120/187613055-e7181fbb-396f-4f45-bd24-5df0ad8854bb.png">

AUC is 0.50, the model is not doing greater than logistic regression. 

#### Random Forest

Random Forest is an ensemble method and built on the idea of bagging. The logic of ensemble method is to combine multiple weaker learner, and create a stronger learner. <br>
<br>
One of the avantages of random forest is it generates decision trees that are uncorrected to promote diversity among trees because random forest makes a set of decision trees.

<img width="700" alt="image" src="https://user-images.githubusercontent.com/64395120/187613548-1aa7b9e2-71a4-4894-bc44-63cf2ea0ee3e.png">

This model is doing slightly better than the previous models; since the AUC (0.53) is greater. However, we are still looking for AUC value closer to 1.0, random forest is not an optimal choice.

#### Gradient Boosting Ensemble

Another ensemble method, Gradient Boosting Ensemble, which is adaptively chaning distribution of training data by focusing more on previously misclassified records. <br>
<br>
Gradient Boosting Emsemble produces the same AUC = 0.53 as Random Forest. 
<img width="700" alt="image" src="https://user-images.githubusercontent.com/64395120/187613760-c5c5d0fb-99d8-4d83-9e0e-0e7ee23d7e05.png">

### Deep Learning
Tensorflow and Keras will be using here to create a small neural network. <br>

We tested 50 epochs and the model stops at epoch 11. 
![Screen Shot 2022-08-31 at 2 03 48 AM](https://user-images.githubusercontent.com/64395120/187614594-e29c69aa-f395-4a64-af32-8ed86f4c2eef.png)

AUC plot: 
![Screen Shot 2022-08-31 at 2 04 25 AM](https://user-images.githubusercontent.com/64395120/187614726-b6581309-dd09-4109-9edb-b47d45f1066b.png)

AUC does not improve, so ANN is not our choice. 

###  Conclusion
Based on the AUC value, the winner here is Random Forest and Gradient Boosting Ensemble with AUC of 0.53. <br>
<br>
The factors that have higher impact on the stock price prediction are Price Range, Moving Average 7, Price Direction, Volume, and lastly Moving Average 3. <br>

![Screen Shot 2022-08-31 at 2 06 07 AM](https://user-images.githubusercontent.com/64395120/187614991-380be9d0-1b2c-492b-a025-c0549bbf8f93.png)



