# PR Project - Internet firewall Action Classification and Prediction
### This is the project in which we have trained a model to classify and predict the Internet firewall action, and this classification and prediction is based on the data-setcollected from the ‘UCI machine learning Repository’. Here we are going to split the data into appropriate ratio i.e., 3:1, and first part is used to train the model and another part is to test the model. According to the data we can say that it’s multiclass classification, there are four actions given in the data-set, which can be understood as four classes.
# Data-set Information
### No. of Attributes = 12
### No. of Instances >= 1000
### Attributes are –
    Source Port
    Destination Port
    NAT Source Port
    NAT Destination Port
    Action – Target Attribute
    Bytes
    Bytes Sent
    Bytes Received
    Packets
    Elapsed Time (sec)
    pkts_sent
    pkts_received
# Random Forest Classifier
### Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction.

<p align = "center">
  <img src="https://user-images.githubusercontent.com/56586584/144798799-3fb538b2-32a8-4968-9bd8-1e344b375bd6.jpg" alt="Random forest Classifier">
</p>

### The fundamental concept behind random forest is a simple but powerful one — the wisdom of crowds. In data science speak, the reason that the random forest model works so well is:
## 'A large number of relatively uncorrelated models (trees) operating as a committee will outperform any of the individual constituent models.'
# Implementation
#### First, we tried to read .csv data-set file, and understood the basic information like shape, info (to find any missing vale is present or not).
#### As the data do not have any type of missing values, we need not to fill any of the places in data-set.
#### As we know that Action is the target attribute, we found no. of unique values in that Action attribute and plotted the count of each unique Action.

<p align = "center">
  <img src="https://user-images.githubusercontent.com/56586584/144799603-a38ae499-2bfd-479f-8722-af940f023a81.png" alt="Action Count">
</p>

#### After this we divided the data-set into two part, they are ‘num_features’, ‘cat_features’ and understood the spread of values for each attribute.
    Cat_features – [‘Source Port’, ‘Destination Port’, ‘NAT Source Port’, ‘NAT Destination Port’]
    Num_features – All remaining attributes
#### And also, we have done plotting between the Source Port and Destination Port for each Action.
#### Similarly, we also done the plotting between NAT Source Port and NAT Destination Port.

![S_vs_D](https://user-images.githubusercontent.com/56586584/144800046-5e06e2c6-ccb7-4721-8ce5-78da17f94a9f.png)
![NS_vs_ND](https://user-images.githubusercontent.com/56586584/144800060-7fdd628d-0791-48c4-ae83-445b1b70c3b6.png)

#### Now, we again divided whole data-set into four part,
    X_train - data used to train the model.
    Y_train - Action values used for training model.
    X_test - data used for testing of model.
    Y_test - Action values for comparing with y_pred.
#### As this is multi-class problem we have used Random Forest Classifier to classify and predict the values (to get high accuracy).
#### After training and testing we have plotted the confusion matrix i.e., between Actual Actions and Predicted Actions.
#### From this confusion matrix we have calculated the accuracy percentage,
    Accuracy percentage we have acquired is – 99.78
    
<p align = "center">
  <img src="https://user-images.githubusercontent.com/56586584/144800332-8ef55f70-6b95-4c0f-b33a-cf620296ab76.png" alt="Confusion_PR">
</p>

# References
#### https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2
#### https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
