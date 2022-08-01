# Tweets-Recommendation-System
![image](https://user-images.githubusercontent.com/107008585/182155776-84c88cdc-c6e8-40ee-be8d-ee14ba64774e.png)
# DTI5125[EG] Data Science Applications Project Report
Personalised recommendations based on user’s tweets

# Problem formulation
The project aims to identify users' interests by classifying their tweets into categories; eg: sports, entertainment, politics etc. And based on that, recommend other tweets that the users are more likely interested in.
Project outline: 

 ![image](https://user-images.githubusercontent.com/107008585/182066020-b46126f4-fea9-4968-9cec-dfa8fa28b73d.png)
 
# Libraries
	# matplotlib
	# pandas
	# numpy
	# sklearn
	# nltk
	# regex
	# wordcloud
	# yellowbrick
  
# Classification models
We applied heterogeneous ensemble learning on the following classifiers and calculated the final label results using soft voting
Support vector Machine
Naive Bayes classifier
Random Forest. 
We applied 5-fold cross validation on this model while training. Average accuracy 80.32%
we tested the final classification model on the test data. Accuracy of the model on the test set = 78.83%


# Clustering model
If two users are interested in sports, that does not mean that they are interested in the same topics of sport, that’s why we need to cluster each field to determine which more specific tweets the user prefers.
Steps:
Filter the dataset to get all of the rows that contain sports as a type column in the data set.
Transform the data from categorical data to numeric data so we use BOW to transform.
use PCA as a technique of unsupervised dimensionality reduction to minimize our features to plot the fields clusters.


# Future Work:
Take into consideration, depending historical models such as LSTM, we provide tweets history, in our data, so that we become able to if a user is still interested in a field he no longer does
Make the model ready for Deployment specially for twitter platform 
May be apply association rules to which fields a user may be interested in, in the same time
