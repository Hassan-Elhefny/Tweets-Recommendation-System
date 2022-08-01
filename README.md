# Tweets-Recommendation-System
# DTI5125[EG] Data Science Applications Project Report
Personalised recommendations based on user’s tweets

# Problem formulation
The project aims to identify users' interests by classifying their tweets into categories; eg: sports, entertainment, politics etc. And based on that, recommend other tweets that the users are more likely interested in.
Project outline: 
 ![image](https://user-images.githubusercontent.com/107008585/182066020-b46126f4-fea9-4968-9cec-dfa8fa28b73d.png)
# Data Preparation
Integration
	Tweets classification data set from Kaggle platform ↗ containing labelled data of four categories: (Sports, politics, entertainment, and medicine). The data set contains 1162 rows, 2 columns
	The second data set contains financial tweets for sentiment analysis purposes ↗. It contains 8 columns, only the text column will be used to add the financial category to the first data set. The data set contains 28442 rows. A random sample of 300 rows will be chosen to keep the final data set balanced.
Final data frame:
 ![image](https://user-images.githubusercontent.com/107008585/182066050-5a5460ff-d824-4c64-96cf-609195987909.png)
# Cleaning
Rejex was used to:
	Remove user name from the tweet
	Remove tweets URLs
	Make sure that only English characters and numbers are included 
Text feature engineering:
	Bag of words transformation was applied to the text to use it with the Kmeans classifier.
	PCA feature reduction to two features was applied on the BOW for training and visualization

The following figures show the word cloud of each category:
![image](https://user-images.githubusercontent.com/107008585/182066131-31401ffc-dfb3-4f82-ba04-836a92fbc0f2.png)
  
# Classification models
We applied heterogeneous ensemble learning on the following classifiers and calculated the final label results using soft voting
Support vector Machine
Naive Bayes classifier
Random Forest. 
We applied 5-fold cross validation on this model while training. Average accuracy 80.32%
we tested the final classification model on the test data. Accuracy of the model on the test set = 78.83%
![image](https://user-images.githubusercontent.com/107008585/182066213-bafef2db-a665-41b0-9f9f-00de2884ca6f.png)
![image](https://user-images.githubusercontent.com/107008585/182066229-3fabf410-1ed9-4c2d-a67a-9ae9a0d440fc.png)


# Clustering model
If two users are interested in sports, that does not mean that they are interested in the same topics of sport, that’s why we need to cluster each field to determine which more specific tweets the user prefers.
Steps:
	Filter the dataset to get all of the rows that contain sports as a type column in the data set.
	Transform the data from categorical data to numeric data so we use BOW to transform.
	use PCA as a technique of unsupervised dimensionality reduction to minimize our features to plot the fields clusters
The following figures show each field Silhouette & WSS, clusters, and distances between clusters:
Sports:
![image](https://user-images.githubusercontent.com/107008585/182066293-c4bd34c0-9c16-4685-8dce-38328717ed42.png)
![image](https://user-images.githubusercontent.com/107008585/182066318-e654a1f2-b4e5-459c-9e1c-95c6a242177c.png)
![image](https://user-images.githubusercontent.com/107008585/182066338-4785280b-225e-4748-b210-71cef48e2dc3.png)

Entertainment:
![image](https://user-images.githubusercontent.com/107008585/182066382-45765ae9-2b9e-46d8-a746-979425bfd49c.png)
![image](https://user-images.githubusercontent.com/107008585/182066394-d9ac1722-a3d7-4ad0-945f-137bd22ea860.png)
![image](https://user-images.githubusercontent.com/107008585/182066404-dc5819a3-e697-440b-85ad-88400cdcd38b.png)

Medical:
![image](https://user-images.githubusercontent.com/107008585/182066425-8fa4e4a8-94a4-49ee-b687-8b91bf517ae0.png)
![image](https://user-images.githubusercontent.com/107008585/182066440-b63cd768-60cb-4289-bbc2-879c5b722c31.png)
![image](https://user-images.githubusercontent.com/107008585/182066461-61a71aea-f4a2-4557-b291-ad18ba76211b.png)

Politics:
![image](https://user-images.githubusercontent.com/107008585/182066484-fc47c1df-4c0f-4cff-8a7f-6481723795b4.png)
![image](https://user-images.githubusercontent.com/107008585/182066506-226455c9-b437-4f0a-a811-bc688c03bb78.png)
![image](https://user-images.githubusercontent.com/107008585/182066526-95dfe801-bea4-4458-a651-072be5b6df36.png)

Financial:
![image](https://user-images.githubusercontent.com/107008585/182066547-ad2cfec5-25c2-4cdc-af5f-c3ffba4131a3.png)
![image](https://user-images.githubusercontent.com/107008585/182066561-65f59b72-493e-434f-8769-121a2b7ee054.png)
![image](https://user-images.githubusercontent.com/107008585/182066570-268e73ef-3863-44b9-80c1-f4d0b54beb01.png)


# Error analysis
In this project three approaches of error analysis are used:
	Documents average length in correct predictions VS Documents average length in wrong predictions.
	Fields Distribution in correct predictions VS Fields Distribution in wrong predictions.
	Finally, most frequent words that could be the reason why model had misclassification.
	Docs Avg length
documents length in both wrongly and correctly predicted samples are almost the same as the have nearly the same average value (64). So, there is no problem with documents length
	# Fields Distribution
  ![image](https://user-images.githubusercontent.com/107008585/182066600-1fa58e0d-c71b-4c21-8a3b-d16337b3094c.png)

In wrongly predicted samples, about 55% of them are of the “Medical & Entertainment” fields
  ![image](https://user-images.githubusercontent.com/107008585/182066628-f1f3e50f-cbf9-4f0c-a7b8-5567dd008f7a.png)

In correctly predicted sample this is not the case as samples are uniformly distributed with respect to fields. 
	This means that bias towards “Medical & Entertainment” are not possibly just a chance and could be vital part of the problem.
	Words frequency:
  ![image](https://user-images.githubusercontent.com/107008585/182066704-d08796d3-3a5b-4463-bee0-42ddcd330cab.png)
  ![image](https://user-images.githubusercontent.com/107008585/182066735-0e4deca2-8dac-450e-8286-68182509423b.png)


In the right-bottom fig, we can see the most frequent words in the wrongly classified samples. 
	Most of these words don’t even exist in correct prediction samples. 
	More interesting observation that in correctly-classified docs many of these words don’t exist in wrong predicted-docs, despite the fact that some of them have frequency > 20, while the most frequent word in the wrong-docs has only frequency = 4
  
# Test case
	To test the overall model, we extracted 100 random tweets from the data set with the following parts:(45% sports, 30% politics, 15% entertainment, 10% medical, 0%financial)
	First, the classifier model should successfully determine the user's top interests:
Classifier output:
![image](https://user-images.githubusercontent.com/107008585/182066763-6dd9ddcc-b7ff-4a64-8a5d-bedefb57cc41.png)
![image](https://user-images.githubusercontent.com/107008585/182066798-2eba64df-1f85-44b2-be7e-5c84a9fb9353.png)

So, the test user's top interests are sports (45%) and politics (31%).
	The next step is to Find out the user prefers which sports and politics tweets.
	Recommend similar tweets with the same percentages. 

n=int((count of a field cluster in user tweets)/(total number of user tweets about that topic))×10
Example:
n(4) = int(18/(18+10+9+6+1+1))×10= 4
n(4) = int(10/(18+10+9+6+1+1))×10=2
![image](https://user-images.githubusercontent.com/107008585/182066836-b7b9af07-8623-4497-af67-317e60f4e696.png)
![image](https://user-images.githubusercontent.com/107008585/182066848-93b7ddaf-9889-495f-82b1-32bcb01f9d9f.png)
![image](https://user-images.githubusercontent.com/107008585/182066859-38fb7824-90ea-4f51-a1d2-19cdbdbe40b8.png)
![image](https://user-images.githubusercontent.com/107008585/182066867-7fe5b884-a360-48c8-b707-cddcef2c807e.png)


# Future Work:
	Take into consideration, depending historical models such as LSTM, we provide tweets history, in our data, so that we become able to if a user is still interested in a field he no longer does
	Make the model ready for Deployment specially for twitter platform 
	May be apply association rules to which fields a user may be interested in, in the same time
 


