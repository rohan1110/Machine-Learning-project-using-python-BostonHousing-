MACHINE LEARNING USING PYTHON 

 

 

SUMMER TRAINING REPORT 2020 

 

Submitted in partial fulfillment of the 

 

Requirements for the award of the degree 

 

of 

 

Bachelor of Technology in Electronics and Communication Engineering 

 

By: 

 

Name: ROHAN BANSAL 

 

University Enrollment no.: (41476802817/ECE3/2017) 

 

 

 

Department of Electronics and Communication Engineering 

 

Guru Tegh Bahadur Institute of Technology 

 

Guru Gobind Singh Indraprastha University 

 

Dwarka, New Delhi 

 

Year 2017-2021 

 

 

 


DECLARATION 

 

 

I hereby declare that all the work presented in the Summer Training Report 2020 entitled 
“MACHINE LEARNING USING PYTHON” in the partial fulfillment of the requirement 

for the award of the degree of Bachelor of Technology in Electronics and Communication 

Engineering, Guru Tegh Bahadur Institute of Technology, Guru Govind Singh Indraprastha 

University, New Delhi is an authentic record of my own work. 

 

 

 

 

 

 

 

Name: ROHAN BANSAL 

 

University Enrollment no.: (41476802817/ECE3/2017) 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 


CERTIFICATE 

 

 

 

 

 

 

 

 

 

Verifiable Link: https://www.skyfilabs.com/verify-certificate/60091266 

 

 

 

 

 

 

 

 

 


INDEX 

 

S.NO. 

TITLE 

PAGE NO. 

 

1. 

 

1. MACHINE LEARNING 


 1.1 Overview 

 1.1.1. History of Machine Learning 

 1.1.2. Applications of machine learning 

 1.1.3. Types of machine learning 

 1.2. Types of Output 

 1.3. Data 

 1.3.1 Types of data 

 1.3.2. Data processing pipeline 

 1.3.3. Data processing preparation 

 1.4. Terminologies is used in machine learning 

 1.5. Tools used for machine learning 

 1.5.1. Programming tools 

 1.5.2. Data handling in Python 

 1.5.3. Machine learning libraries 

 

6 - 13 

 

2. 

 

2. Boston housing machine learning model 
2.1. Description of problem 
2.2. Strategy for price prediction 
2.3. Import Libraries 
2.4. Steps involved in solving problem with machine learning 
techniques 





 

14 - 15 

 

3. 

 

3. STEP 1: DATA PREPROCESSING 
3.1. Load Dataset 
3.1.1. Put the Data into Pandas Dataframe 
3.1.2. Generate a target dataset 
3.1.3. Concatenate features 



3.2. Data Visualization 
3.3. Correlation between target and attributes 
3.4. Normalization of BH data 





16 - 22 

 

4. 

 

 

 

4. STEP 2: SPLITTING THE DATASET 
4.1. Splitting Dataset 
4.2. Overfitting and Underfitting 
4.3. Steps To avoid overfitting 





23 - 25 




 

5. 

 

5. LINEAR REGRESSION IN MACHINE LEARNING 
5.1. Overview 
5.2. Types of linear regression 
5.3. Learning in Linear regression 
5.4. Prediction in linear regression 





26 - 27 

 

6. 

 

6. GRADIENT DESCENT IN MACHINE LEARNING 
6.1. Concepts and implementation 
6.2. Steps To compute gradient descent 
6.3. Features of gradient descent 
6.4. Programming logic of gradient descent 
6.4.1. Update function 
6.4.2. Error function 
6.4.3. Gradient Descent function 



6.5. Running Gradient Descent function 





28 - 32 

 

7. 

 

7. VISUALIZATION OF THE LEARNING PROCESS 
7.1. Plot regression line 
7.2. Plot error values 





33 

 

8. 

 

8. STEP 4: MODEL TRAINING VISUALIZATION 
8.1. Initialize the variables 
8.2. Defining the Init function 
8.3. Defining the update function 
8.4. Declare the animated object and Generate the video of the 
animation 





34 - 35 

 

9. 

 

9. STEP5- PREDICTION OF PRICES 
9.1. Steps to be performed 
9.2. Calculate the predicted value 
9.3. Compute MSE 
9.4. Put xtest, ytest and predicted values into a single 
DataFrame 
9.5. Plot the predicted values against the target values 
9.6. Revert normalisation 
9.7. Obtain Predicted Output 





36 - 38 



 

 

 


1. MACHINE LEARNING 
1.1.1. History of Machine Learning 





 

 1.1 Overview 

Machine Learning is the field of study that gives computers the capability to learn without being 
explicitly programmed. 

As it is evident from the name, it gives the computer that makes it more similar to humans: The 
ability to learn. 

 

How it is different from traditional programming 

 

 

 

 

The term Machine Learning was coined by Arthur Samuel in 1959, an American pioneer in the 
field of computer gaming and artificial intelligence and stated that “it gives computers the ability 
to learn without being explicitly programmed”. 

And in 1997, Tom Mitchell gave a “well-posed” mathematical and relational definition that “A 
computer program is said to learn from experience E with respect to some task T and some 
performance measure P, if its performance on T, as measured by P, improves with experience E. 

 

 

 

 

 

 

 

 

 


1.1.2. Applications of machine learning 








Formal definition of machine learning. 

 

 

 

 

 

1.1.3. Types of machine learning 

 

 


 

• Supervised Learning 


 

 

• Unsupervised Learning 


 

 

 

 

 

 

 

 


• Reinforcement Learning 
1.2. Types of Output 





 

 

 

• Regression - It has Limited real-world implementation. 
• Classification - it is used mainly in photos classification 


 

 

 


• Clustering- it is a technique where the sample is placed in a cluster where all the samples 
in cluster are similar to each other. It is supervised and unsupervised type of machine 
learning. it is generally used in anomalies detection and fraud detection 


 

 

 

 1.3. Data 

 

 1.3.1 Types of data 

 

 

 1.3.2. Data processing pipeline 

 

 

 


 1.3.3. Data processing preparation 

 

 

 

 1.4. Terminologies is used in machine learning 

 

a) Features 
● measure property of data object 
● used as inputVariable 
● choosing distinguishing and independent features is crucial 
● example colour size test extra 


 

b) Target 
● Expected output of given feature 
● value to be predicted using machine learning 
● example category/ name of fruit yield of crop 


 

c) Label - one type of target values 


 

d) Model - hypothesis that defines relationship between features and target 


 

e) Training 
● The process layer model use data to learn and to predict the output 
● data set divided into two parts training data set which obtain standardise model ll 
and testing data set that test how how well model performs 
● expose model to feature and expected targets 
● laws relationship between labour and features 
f) Prediction 
● Apply model to unseen data 
● the target / label based on the data 



 

 1.5. Machine Learning Pipeline 

 

 

 

 

1.6. Tools used for machine learning 

 

 1.6.1. Programming tools 


 

 1.6.2. Data handling in Python 

 


 1.6.3. Machine learning libraries 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 


 

2. BOSTON HOUSING MACHINE LEARNING MODEL 
2.1. Description of problem 





 

 

• Prediction of Price of houses in various places in Boston 
• data set has 506 data points 
• 14 columns in which 13 features and one is target price 
2.2. Strategy for price prediction 
2.3. Import Libraries 





 

 

 

 

 

 

 

 


2.4. Steps involved in solving problem with machine learning techniques 





i. Data preprocessing 
ii. Splitting of data in two parts training data set and testing data set 
iii. Define Error 
iv. Train the Model 
v. Prediction 


 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 


3. STEP 1: DATA PREPROCESSING 
3.1. Load Dataset 





 

 

 

 

 

 

 

 

 

 

 


3.1.1. Put the Data into Pandas Dataframe 
3.1.2. Generate a target dataset 








 

 

 

 

 

 


3.1.3. Concatenate features and target into a single DataFrame and print it 



3.2. Data Visualization of BH dataset 





 

 

 

 

Description is usually the first step that we must perform after importing the data to gather basic 
insights regarding the data so that we can understand reach off of each variable. Also to 
understand the overview of distribution of variable values use describe to generate a summary of 
data set. 

‘Dataset.describe()’ function generate descriptive stats that summaries the central tendency 
dispersion and shape of data set distribution. 

The describe() method computeS the following parameters for each column: 

• count - number of rows 
• mean - mean of column 
• std - standard deviation in column 



• max -maximum value in column 
• min - minimum value of in Kollam 
• 25% - 25 percentile 
• 50% - 50 percentile 
• 75% - 75 percentile 
3.3. Correlation between target and attributes 





Percentile or centile is a measure used in statistics indicating the value below which a given 
percentage of observations in a group of observation Falls 

 

 

 

 In order to perform linear regression, we want to know which of the feature can be used to 
predict the target variable there are multiple techniques available to implement this we will do 
this by exam meaning the statistical metric called correlation. 

Correlation describes how closely the value in are dependent on the values of other column 
basically it is a relationship between two columns . 

 

 

Positive Correlation- 

• change in values of y also shows similar change in values of x 
• if x increases y increases 
• if x decreases by decreases 


 

 


 Negative correlation 

• if x increases by decreases 
• if x decreases by increases 


 

 Whichever attribute has higher absolute correlation with target that is the attribute we will 
choose as the independent variable to perform linear regression 

 Note : an existing of correlation between two variables doesn't mean linear dependence 
between the variable 

 

 

 

• We observed from the bar graph above, that LSTAT and RM features have highest 
absolute correlation value with the target. 



 

• If we take Correaltion between Target and Attribures without taking absolute values 


 

• We observed from the bar graph above, that LSTAT and RM features have highest 
absolute correlation value with the target but LSTAT has negative value and RM has 
positive. 
3.4. Normalization of BH data 





 

Goal of normalization is to change the values such that after the transformation all the values like 
in a common scale that is 0 to 1 in our case. 

Without normalisation it is sometimes difficult to interpret the data. 

Normalisation brings all the values in a common scale without distorting the values the data 
becomes easier to interpret. 

 

Before normalisation values are in arbitrary range 

“sklearn” library provides MinMaxScaler method that takes a list of values in any arbitrary range 
and give them in a list in which values are between 0 to 1. 

 

 MinMaxScaler object stores the parameter required to normalise the values therefore we have to 
use separate scalar objects to perform scaling of different columns so that it we can then use to 
stored parameters to obtain the scaled values in original representation. 

MinMaxScaler provides a method called inverse transform to obtain the values in original 
representation so that we can compare the predicted values and True values. 

 

The fit transform function computes the minimum and maximum, transform the values and 
return normalised values. expects values column wise instead of horizontal it has to be vertical. 

 

 

 

 


Therefore she shape values with numpy function reshape to have Values in one column. 

We do not have to know it before and we can pass -1 for one of the dimensions and correct value 
is used for the dimension now pass the reshaped values as a parameter to the fit transform 
method and get normalised value stored in the variable x 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 


 

 

 

4. STEP 2: SPLITTING THE DATASET 
4.1. Splitting Dataset 





 

 

 

The model is initially fitted on training data set that is a set of examples used in in to ft the 
parameters of the model. the model is trained on training data set using a supervised learning 
method. 

In practice the training data set consists of pair of set of features and corresponding target label. 

 the current model Run with training data set and produces a result which is then compared with 
the target. based on the result of comparison and specific learning algorithm being used the 
parameters of the model are adjusted. 

The fitted model is used to make prediction based on second data set called validation data set. it 
provides an unbiased evaluation of a model fit on training data set by tuning the model. hence 
the model occasionally sees this data but never does it learn from this. we use validation data set 
result and update higher level parameters of model. it affects the model but in a indirect way. 

Test data set is used to provide unbiased evaluation of the final model with on the train data set. 
it is only used once the model is fully trained using training and validation data set. 

 

The training set contains on own output and the model long from this data in order to be 
generalized to other data later on. 

 We have the test data set in order to test our models prediction on the subset 

 

 Training data set is usually divided in 2 parts- validation data set which usually is 10 to 15% of 
the total data set and training data set which is the rest. 


4.2. Overfitting and Underfitting 





 

 

 Overfitting 

● When a model Learns the detail and noise in the training data set to the extent that it 
negatively impact the performance of the model to the new data 
● model that fits the data too well 
● Results in excessively complicated model 
● doesn't apply to new data 


 Underfitting 

● When the machine learning algorithm can capture the trend of the data 
● model doesn't fill well enough 
● often results in a accessibility simple model 


 

 

 4.3. Steps To avoid overfitting 

a) using validation 
b) fitting multiple models 
c) using Cross Validation- 
● Similar to train test split but it is applied to more subsets 



● meaning that we split our data into k subsets and train ke -1 subset and we hold last 
subset for test data set 


 

 Split Testing data set randomly choosing certain percentage of samples from the data set and 
exclude them from the data shown them during training phase that is training data 

 Note important to choose randomly to ensure that data is uniformly distributed and we don't 
introduce any bias case. 

 

‘sklearn’ provides a function called train test split 

 the parameter to this function are data set and fraction of data set to be considered for the testing 
data set 

 It will return four values- 

• training set of features X train 
• Testing set of features X test 
• training set of target by train 
• testing set of target buy test 


 

 

 

 

 

 

 

 

 

 


5. LINEAR REGRESSION IN MACHINE LEARNING 
5.1. Overview 
5.2. Types of linear regression 





 

• Origin in statistics 
• understand relationship between input and output numerical variable 
• supervised learning 
• predicted output is continuous rather than discrete 


 

 

 

 

 

 

 

 

 


5.3. Learning in Linear regression 
5.4. Prediction in linear regression 





 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 


6. GRADIENT DESCENT IN MACHINE LEARNING 
6.1. Concepts and implementation 





 

 

• Optimising algorithm to minimise function 
● iteratively move in the direction of steepest descent 
● steepest descent - largest negative gradient 
● impossible to visualise gradient in a space consisting more than three dimensions 
• too low - require many steps although accurate 
• Too high - might lead to divergence 





 

 

 

Learning rate- size of iterative step 

 

 Cost function- the function to be minimized in machine learning 

 

 

 

 

 

 

 


6.2. Steps To compute gradient descent 

 

 

6.3. Features of gradient descent 

 

 

 


 

 

 6.4. Programming logic of gradient descent 

 there are three functions which have to be defined 

● Update function 
● error function 
● gradient Descent function 


 

6.4.1. Update function 

 

 

 

 

 

 

 




 

6.4.2. Error function 

 

 

 

 

6.4.3. Gradient Descent function 

 

 


• Learning rate- Increasing the learning rate reduces the convergence time but if the 
learning rate is too high the model will overshoot the minima for this problem set 
the value below 0.0025 otherwise it will cause overflow in weight values 
• Iterations- number of iterations must be large enough to allow the model to 
converge to a minima but if it's too large then the model become too specific to 
the training data thus causing overfitting that the model memorizes the data 
instead of burning the data 
• Errors threshold- this value can be set to a maximum value of error that is 
acceptable when the error values goes below the threshold the gradient Descent is 
stopped 
• Initial values for this problem where our objective is to determine the line which 
gives the least error and does not matter what the initial values you provide but 
for non convex Optimisation problems initial value affects the learning rate. 





 

 

6.5. Running Gradient Descent function 

 

First we have to define the Hyperparameters. 

Hyper parameters are the parameters that may change and varied to observe the 
computation versus accuracy traders 

 

 

 


7. VISUALIZATION OF THE LEARNING PROCESS 


 

Use error values array to plot the values. 

This is to observe how error changes during training model is getting better the error 
values must fall over regression line 

i. Plot Regression line- against the training data set to visualize what the line looks 
like for the training data set. 


 

 

ii. Plot Error Values- shows how the error drops over time 


 

 


 

8. STEP 4: MODEL TRAINING VISUALIZATION 
8.1. Initialize the variables 
8.2. Defining the Init function 





 

As the number of iterations increases, changes in the line are less noticable 

In order to reduce the processing time for the animation, it is advised to choose values 

 

 

 

 

 

 

X and Y coordinates are empty list 

 generating and animation requires to function Ban to initialise the state of the graph and other to 
update each function with new data 

 update function is called before drawing each frame 

 inside the update function we will send new endpoints for the line 

 

 

 

 


8.3. Defining the update function 
8.4. Declare the animated object and Generate the video of the animation 





 

 

 

 

 

 


9. STEP5- PREDICTION OF PRICES 
9.1. Steps to be performed 
9.2. Calculate the predicted value on the test set as a vectorized operation 
9.3. Compute MSE for the predicted values on the training set 
9.4. Put xtest, ytest and predicted values into a single DataFrame , so that 
we can see the predicted values alongside the testing data 





 

 

 

 

 

 

 

 

 


9.5. Plot the predicted values against the target values 





 

• Predicted values are represented by red colour line 
• Target values are represented by blue colour line 


 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 


9.6. Revert normalisation to obtain the predicted price of house in 1000 
dollars. 





And find the final prediction prices 

The predicted value are in range of 0-1. This is not very useful to us when we want to obtain the 
price. Use inverse transform to scale the values back to the original representation. 

 

 

 

 

 

 

 

 

 

 

 

 

 


 

 

REFERENCES 

 

• www.skyfilabs.com 


 

 

 

Github link for Project code- 

 

https://github.com/rohan1110/Machine-Learning-project-using-python-
BostonHousing- 

 


