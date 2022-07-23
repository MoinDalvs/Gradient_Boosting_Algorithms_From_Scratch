## Quick Introduction to Boosting (What is Boosting?)
### Picture this scenario:

You’ve built a linear regression model that gives you a decent 77% accuracy on the validation dataset. Next, you decide to expand your portfolio by building a k-Nearest Neighbour (KNN) model and a decision tree model on the same dataset. These models gave you an accuracy of 62% and 89% on the validation set respectively.

It’s obvious that all three models work in completely different ways. For instance, the linear regression model tries to capture linear relationships in the data while the decision tree model attempts to capture the non-linearity in the data.
![image](https://user-images.githubusercontent.com/99672298/180591154-566b9fb9-14f7-403c-abe7-f280cb18a0c5.png)\
How about, instead of using any one of these models for making the final predictions, we use a combination of all of these models?

I’m thinking of an average of the predictions from these models. By doing this, we would be able to capture more information from the data, right?

That’s primarily the idea behind ensemble learning. And where does boosting come in?

Boosting is one of the techniques that uses the concept of ensemble learning. A boosting algorithm combines multiple simple models (also known as weak learners or base estimators) to generate the final output.
## 1. Gradient Boosting Machine (GBM)
A Gradient Boosting Machine or GBM combines the predictions from multiple decision trees to generate the final predictions. Keep in mind that all the weak learners in a gradient boosting machine are decision trees.

But if we are using the same algorithm, then how is using a hundred decision trees better than using a single decision tree? How do different decision trees capture different signals/information from the data?

Here is the trick – the nodes in every decision tree take a different subset of features for selecting the best split. This means that the individual trees aren’t all the same and hence they are able to capture different signals from the data.

Additionally, each new tree takes into account the errors or mistakes made by the previous trees. So, every successive decision tree is built on the errors of the previous trees. This is how the trees in a gradient boosting machine algorithm are built sequentially.
![image](https://user-images.githubusercontent.com/99672298/180591175-9472eed5-f0f1-4fb7-9dce-0b06bee28f12.png)\
We already know that errors play a major role in any machine learning algorithm. There are mainly two types of error, bias error and variance error. Gradient boost algorithm helps us minimize bias error of the model.
### The first realization of boosting that saw great success in application was Adaptive Boosting or AdaBoost for short.
AdaBoost Algorithm which is again a boosting method. The weak learners in AdaBoost are decision trees with a single split, called decision stumps for their shortness. This algorithm starts by building a decision stump and then assigning equal weights to all the data points. Then it increases the weights for all the points which are misclassified and lowers the weight for those that are easy to classify or are correctly classified. A new decision stump is made for these weighted data points. The idea behind this is to improve the predictions made by the first stump. New weak learners are added sequentially that focus their training on the more difficult patterns.The main difference between these two algorithms is that Gradient boosting has a fixed base estimator i.e., Decision Trees whereas in AdaBoost we can change the base estimator according to our needs.
## What is boosting?
While studying machine learning you must have come across this term called Boosting. It is the most misinterpreted term in the field of Data Science. The principle behind boosting algorithms is first we built a model on the training dataset, then a second model is built to rectify the errors present in the first model. Let me try to explain to you what exactly does this means and how does this works.
![image](https://user-images.githubusercontent.com/99672298/180592033-ba34e5cd-e22c-4a6b-a4b6-3588662a0cfb.png)\
Suppose you have n data points and 2 output classes (0 and 1). You want to create a model to detect the class of the test data. Now what we do is randomly select observations from the training dataset and feed them to model 1 (M1), we also assume that initially, all the observations have an equal weight that means an equal probability of getting selected.

Remember in ensembling techniques the weak learners combine to make a strong model so here M1, M2, M3….Mn all are weak learners.

Since M1 is a weak learner, it will surely misclassify some of the observations. Now before feeding the observations to M2 what we do is update the weights of the observations which are wrongly classified. You can think of it as a bag that initially contains 10 different color balls but after some time some kid takes out his favorite color ball and put 4 red color balls instead inside the bag. Now off-course the probability of selecting a red ball is higher. This same phenomenon happens in Boosting techniques, when an observation is wrongly classified, its weight get’s updated and for those which are correctly classified, their weights get decreased. The probability of selecting a wrongly classified observation gets increased hence in the next model only those observations get selected which were misclassified in model 1.

Similarly, it happens with M2, the wrongly classified weights are again updated and then fed to M3. This procedure is continued until and unless the errors are minimized, and the dataset is predicted correctly. Now when the new datapoint comes in (Test data) it passes through all the models (weak learners) and the class which gets the highest vote is the output for our test data.
### Understand Gradient Boosting Algorithm with example
Let’s understand the intuition behind Gradient boosting with the help of an example. Here our target column is continuous hence we will use Gradient Boosting Regressor.

Following is a sample from a random dataset where we have to predict the car price based on various features. The target column is price and other features are independent features.

![image](https://user-images.githubusercontent.com/99672298/180592447-05d51d72-bd76-40b2-850f-da745d8e0e75.png)\

#### Step -1 The first step in gradient boosting is to build a base model to predict the observations in the training dataset. For simplicity we take an average of the target column and assume that to be the predicted value as shown below:

![image](https://user-images.githubusercontent.com/99672298/180592468-df49c744-2394-4254-b90f-63809377f4fb.png)

Looking at this may give you a headache, but don’t worry we will try to understand what is written here.

Here L is our loss function

Gamma is our predicted value

argmin means we have to find a predicted value/gamma for which the loss function is minimum.

Since the target column is continuous our loss function will be:

![image](https://user-images.githubusercontent.com/99672298/180592483-f1f2c325-649a-4e1e-b866-949fb111a529.png)\

loss function | Gradient Boosting Algorithm
Here yi is the observed value

And gamma is the predicted value

Now we need to find a minimum value of gamma such that this loss function is minimum. We all have studied how to find minima and maxima in our 12th grade. Did we use to differentiate this loss function and then put it equal to 0 right? Yes, we will do the same here.

![image](https://user-images.githubusercontent.com/99672298/180592494-7fdb2c75-654f-45a9-b182-1ac654c43747.png)

differentiate loss function
Let’s see how to do this with the help of our example. Remember that y_i is our observed value and gamma_i is our predicted value, by plugging the values in the above formula we get:

![image](https://user-images.githubusercontent.com/99672298/180592500-c818dd45-37f4-492f-a639-715eb4cf0bba.png)

plug values | Gradient Boosting Algorithm
We end up over an average of the observed car price and this is why I asked you to take the average of the target column and assume it to be your first prediction.

Hence for gamma=14500, the loss function will be minimum so this value will become our prediction for the base model.
![image](https://user-images.githubusercontent.com/99672298/180592508-eb40a933-f93a-401c-b225-fe751fd84807.png)
#### Step-2 The next step is to calculate the pseudo residuals which are (observed value – predicted value)

Again the question comes why only observed – predicted? Everything is mathematically proved, let’s from where did this formula come from. This step can be written as:

![image](https://user-images.githubusercontent.com/99672298/180592566-be077eb8-3843-4735-bf10-4269b26fd5e0.png)

Here F(xi) is the previous model and m is the number of DT made.

The predicted value here is the prediction made by the previous model. In our example the prediction made by the previous model (initial base model prediction) is 14500, to calculate the residuals our formula becomes:

![image](https://user-images.githubusercontent.com/99672298/180592632-e80350f1-b5f7-4239-99d3-1f69af26087d.png)
![image](https://user-images.githubusercontent.com/99672298/180592557-6acc1beb-8907-4353-af6f-8ddb627f0055.png)

In the next step, we will build a model on these pseudo residuals and make predictions. Why do we do this? Because we want to minimize these residuals and minimizing the residuals will eventually improve our model accuracy and prediction power. So, using the Residual as target and the original feature Cylinder number, cylinder height, and Engine location we will generate new predictions. Note that the predictions, in this case, will be the error values, not the predicted car price values since our target column is an error now.

Let’s say hm(x) is our DT made on these residuals.
#### Step- 4 In this step we find the output values for each leaf of our decision tree. That means there might be a case where 1 leaf gets more than 1 residual, hence we need to find the final output of all the leaves. TO find the output we can simply take the average of all the numbers in a leaf, doesn’t matter if there is only 1 number or more than 1.\
Let’s see why do we take the average of all the numbers. Mathematically this step can be represented as:

![image](https://user-images.githubusercontent.com/99672298/180592730-8c4444e7-79e1-4f5f-a5b6-22e04c6c0a41.png)

Here hm(xi) is the DT made on residuals and m is the number of DT. When m=1 we are talking about the 1st DT and when it is “M” we are talking about the last DT.

The output value for the leaf is the value of gamma that minimizes the Loss function. The left-hand side “Gamma” is the output value of a particular leaf. On the right-hand side [Fm-1(xi)+ƴhm(xi))] is similar to step 1 but here the difference is that we are taking previous predictions whereas earlier there was no previous prediction.

Let’s understand this even better with the help of an example. Suppose this is our regressor tree:

![image](https://user-images.githubusercontent.com/99672298/180592739-c1ad662e-81a4-45db-95e1-690e5383f617.png)

We see 1st residual goes in R1,1  ,2nd and 3rd residuals go in R2,1 and 4th residual goes in R3,1 .

Let’s calculate the output for the first leave that is R1,1

![image](https://user-images.githubusercontent.com/99672298/180592751-4a6799b5-04e3-4898-95bd-3f95b87ec836.png)

Now we need to find the value for gamma for which this function is minimum. So we find the derivative of this equation w.r.t gamma and put it equal to 0.

![image](https://user-images.githubusercontent.com/99672298/180592757-40d1952d-d41e-4801-8c75-7411fdcdd00e.png)

Hence the leaf R1,1 has an output value of -2500. Now let’s solve for the R2,1

![image](https://user-images.githubusercontent.com/99672298/180592813-221232e1-55c8-416c-9e55-cc7231d01ae8.png)

Let’s take the derivative to get the minimum value of gamma for which this function is minimum:

![image](https://user-images.githubusercontent.com/99672298/180592825-83b97463-e9d2-4a92-8009-51139455f696.png)

We end up with the average of the residuals in the leaf R2,1 . Hence if we get any leaf with more than 1 residual, we can simply find the average of that leaf and that will be our final output.

Now after calculating the output of all the leaves, we get:

![image](https://user-images.githubusercontent.com/99672298/180592833-a5b59e63-bd97-4da3-8b41-2d76155690d7.png)

#### Step-5 This is finally the last step where we have to update the predictions of the previous model. It can be updated as:
![image](https://user-images.githubusercontent.com/99672298/180592838-6d150e8f-9cd1-4b1c-a9b7-defced68e81b.png)

where m is the number of decision trees made.

Since we have just started building our model so our m=1. Now to make a new DT our new predictions will be:

![image](https://user-images.githubusercontent.com/99672298/180592844-6b95fe59-f048-49f7-8ff9-3bb64b6fa3d9.png)

Here Fm-1(x) is the prediction of the base model (previous prediction) since F1-1=0 , F0 is our base model hence the previous prediction is 14500.

nu is the learning rate that is usually selected between 0-1. It reduces the effect each tree has on the final prediction, and this improves accuracy in the long run. Let’s take nu=0.1 in this example.

Hm(x) is the recent DT made on the residuals.

Let’s calculate the new prediction now:

![image](https://user-images.githubusercontent.com/99672298/180592852-56692fcf-636a-41d1-aaee-0f16474df415.png)


