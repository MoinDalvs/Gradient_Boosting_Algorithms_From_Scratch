## 0.1 Table of Contents<a class="anchor" id="0.1"></a>
1. [Quick Introduction to Boosting (What is Boosting?)](#1)
    - 1.1 [Gradient Boosting Machine (GBM)](#1.1)
    - 1.2 [What is boosting?](#1.2)
    - 1.3 [Improvements to Basic Gradient Boosting](#1.3)
    - 1.4 [Summary](#1.4)
    - 1.5 [Maths Intuition (Regression)](#1.5)
    - 1.6 [Maths Intuition (Classification)](#1.6)
2. [XGBM (Extreme Gradient Boosting Machine)](#2)
    - 2.1 [Missing Values](#2.1)
3. [Exploratory Data Analysis](#3)
    - 3.1 [Outlier Detection](#3.1)
4. [Data Visualization](#4)
5. [Data Pre-processing](#5)
    - 5.1 [Standardizing the Data](#5.1)
    - 5.2 [Normalizing the Data](#5.2)
6. [KMeans Clustering](#6)
    - 6.1 [Elbow Method for Determining Cluster Amount on Standard Scaled Data](#6.1)
    - 6.2 [Silhoutte Score](#6.2)
    - 6.3 [Build KMeans Cluster algorithm using K=6 and Standard Scaler Applied Dataset](#6.3)
    - 6.4 [Elbow Method and Silhouette Score on MinMaxScaler Applied Data](#6.4)
    - 6.5 [Build KMeans Cluster algorithm using K=4 and MinMaxScaler Applied Dataset](#6.5)
    - 6.6 [t-SNE Visualization](#6.6)
    - 6.6 [UMAP Visualization](#6.6)
7. [Hierarchical Clustering Algorithm](#7)
    - 7.1 [Dendogram on MinMaxScaler Applied on Dataset](#7.1)
    - 7.2 [Dendrogram on Standard Scaler Applied on Data](#7.2)
    - 7.3 [PCA](#7.3)
        - 7.3A [Running PCA of Standardized data](#7.3A)
        - 7.3B [Silhouette Score method for PCA Standard Scaled Data](#7.3B)
        - 7.3C [Run Hierarchical Clustering.(Agglomerative Clustering) on Standard Scaled Data](#7.3C)
        - 7.3D [Running PCA of Normalized data](#7.3D)
        - 7.3E [Silhouette Score method for PCA MinMax Scaled Data](#7.3E)
        - 7.3F [Run Hierarchical Clustering.(Agglomerative Clustering) on MinMax Scaled Data](#7.3F)
8. [DBSCAN - (Density Based Spatial Clustering of Applications with Noise)](#8)
    - 8.1 [DBSCAN of Standard Scaled Data](#8.1)
    - 8.2 [DBSCAN of MinMax Scaled Data](#8.2)
9. [Conclusion](#9)


## 1) Quick Introduction to Boosting (What is Boosting?)<a class="anchor" id="1"></a>
### Picture this scenario:

You’ve built a linear regression model that gives you a decent 77% accuracy on the validation dataset. Next, you decide to expand your portfolio by building a k-Nearest Neighbour (KNN) model and a decision tree model on the same dataset. These models gave you an accuracy of 62% and 89% on the validation set respectively.

It’s obvious that all three models work in completely different ways. For instance, the linear regression model tries to capture linear relationships in the data while the decision tree model attempts to capture the non-linearity in the data.
![image](https://user-images.githubusercontent.com/99672298/180591154-566b9fb9-14f7-403c-abe7-f280cb18a0c5.png)\
How about, instead of using any one of these models for making the final predictions, we use a combination of all of these models?

I’m thinking of an average of the predictions from these models. By doing this, we would be able to capture more information from the data, right?

That’s primarily the idea behind ensemble learning. And where does boosting come in?

Boosting is one of the techniques that uses the concept of ensemble learning. A boosting algorithm combines multiple simple models (also known as weak learners or base estimators) to generate the final output.
## 1.1 Gradient Boosting Machine (GBM)<a class="anchor" id="1.1"></a>

[Table of Content](#0.1)

A Gradient Boosting Machine or GBM is an ensemble machine learning algorithm that can be used for classification or regression predictive modeling problems, which combines the predictions from multiple decision trees to generate the final predictions. Keep in mind that all the weak learners in a gradient boosting machine are decision trees. The main objective of Gradient Boost is to minimize the loss function by adding weak learners using a gradient descent optimization algorithm. The generalization allowed arbitrary differentiable loss functions to be used, expanding the technique beyond binary classification problems to support regression, multi-class classification, and more.

The models that form the ensemble, also known as base learners, could be either from the same learning algorithm or different learning algorithms. Bagging and boosting are two widely used ensemble learners. Though these two techniques can be used with several statistical models, the most predominant usage has been with decision trees.

But if we are using the same algorithm, then how is using a hundred decision trees better than using a single decision tree? How do different decision trees capture different signals/information from the data?
#### Bagging
While decision trees are one of the most easily interpretable models, they exhibit highly variable behavior. Consider a single training dataset that we randomly split into two parts. Now, let’s use each part to train a decision tree in order to obtain two models.

When we fit both these models, they would yield different results. Decision trees are said to be associated with high variance due to this behavior. Bagging or boosting aggregation helps to reduce the variance in any learner. Several decision trees which are generated in parallel, form the base learners of bagging technique. Data sampled with replacement is fed to these learners for training. The final prediction is the averaged output from all the learners.

Here is the trick – the nodes in every decision tree take a different subset of features for selecting the best split. This means that the individual trees aren’t all the same and hence they are able to capture different signals from the data.

![21 07 2022_12 45 18_REC](https://user-images.githubusercontent.com/99672298/180593714-ed5f2021-1aa2-4f30-861b-86af44fb1980.png)

#### Boosting
Additionaly in boosting, the trees are built sequentially such that each subsequent tree aims to reduce the errors of the previous tree. Each tree learns from its predecessors and updates the residual errors. Hence, the tree that grows next in the sequence will learn from an updated version of the residuals.

The base learners in boosting are weak learners in which the bias is high, and the predictive power is just a tad better than random guessing. Each of these weak learners contributes some vital information for prediction, enabling the boosting technique to produce a strong learner by effectively combining these weak learners. As we already know that errors play a major role in any machine learning algorithm. There are mainly two types of error, bias error and variance error. The final strong  learner helps us minimize bring down both the bias and the variance.

In contrast to bagging techniques like Random Forest, in which trees are grown to their maximum extent, boosting makes use of trees with fewer splits. Such small trees, which are not very deep, are highly interpretable. Parameters like the number of trees or iterations, the rate at which the gradient boosting learns, and the depth of the tree, could be optimally selected through validation techniques like k-fold cross validation. Having a large number of trees might lead to overfitting. So, it is necessary to carefully choose the stopping criteria for boosting.

![image](https://user-images.githubusercontent.com/99672298/180591175-9472eed5-f0f1-4fb7-9dce-0b06bee28f12.png)\

### The first realization of boosting that saw great success in application was Adaptive Boosting or AdaBoost for short.
AdaBoost Algorithm which is again a boosting method. The weak learners in AdaBoost are decision trees with a single split, called decision stumps for their shortness. This algorithm starts by building a decision stump and then assigning equal weights to all the data points. Then it increases the weights for all the points which are misclassified and lowers the weight for those that are easy to classify or are correctly classified. A new decision stump is made for these weighted data points. The idea behind this is to improve the predictions made by the first stump. New weak learners are added sequentially that focus their training on the more difficult patterns.The main difference between these two algorithms is that Gradient boosting has a fixed base estimator i.e., Decision Trees whereas in AdaBoost we can change the base estimator according to our needs.

AdaBoost uses multiple iterations to generate a single composite strong learner. It creates a strong learner by iteratively adding weak learners. During each phase of training, a new weak learner is added to the ensemble, and a weighting vector is adjusted to focus on examples that were misclassified in previous rounds. The result is a classifier that has higher accuracy than the weak learner classifiers.

![image](https://user-images.githubusercontent.com/99672298/180611411-faf62bc8-9795-444d-85f4-3a6ffab8cec8.png)

Gradient Boosting trains many models in a gradual, additive and sequential manner. The major difference between AdaBoost and Gradient Boosting Algorithm is how the two algorithms identify the shortcomings of weak learners (eg. decision trees). Thus, like AdaBoost, Gradient Boost builds fixed sized trees based on the previous tree's errors, but unlike AdaBoost, each tree can be larger than a stump.In Contrast, Gradient Boost starts by making a single leaf, instead of a tree or a stump. While the AdaBoost model identifies the shortcomings by using high weight data points, gradient boosting performs the same by using gradients in the loss function

## 1.2 What is boosting?<a class="anchor" id="1.2"></a>

While studying machine learning you must have come across this term called Boosting. Boosting is an ensemble learning technique to build a strong classifier from several weak classifiers in series. Boosting algorithms play a crucial role in dealing with bias-variance trade-offs. Unlike bagging algorithms, which only control for high variance in a model, boosting controls both the aspects (bias & variance) and is considered to be more effective.

Below are the few types of boosting algorithms:

1. AdaBoost (Adaptive Boosting)
2. Gradient Boosting
3. XGBoost
4. CATBoost
5. Light GBM

The principle behind boosting algorithms is first we built a model on the training dataset, then a second model is built to rectify the errors present in the first model. Let me try to explain to you what exactly does this means and how does this works.
![image](https://user-images.githubusercontent.com/99672298/180592033-ba34e5cd-e22c-4a6b-a4b6-3588662a0cfb.png)\
Suppose you have n data points and 2 output classes (0 and 1). You want to create a model to detect the class of the test data. Now what we do is randomly select observations from the training dataset and feed them to model 1 (M1), we also assume that initially, all the observations have an equal weight that means an equal probability of getting selected.

Remember in ensembling techniques the weak learners combine to make a strong model so here M1, M2, M3….Mn all are weak learners.

Since M1 is a weak learner, it will surely misclassify some of the observations. Now before feeding the observations to M2 what we do is update the weights of the observations which are wrongly classified. You can think of it as a bag that initially contains 10 different color balls but after some time some kid takes out his favorite color ball and put 4 red color balls instead inside the bag. Now off-course the probability of selecting a red ball is higher. This same phenomenon happens in Boosting techniques, when an observation is wrongly classified, its weight get’s updated and for those which are correctly classified, their weights get decreased. The probability of selecting a wrongly classified observation gets increased hence in the next model only those observations get selected which were misclassified in model 1.

Similarly, it happens with M2, the wrongly classified weights are again updated and then fed to M3. This procedure is continued until and unless the errors are minimized, and the dataset is predicted correctly. Now when the new datapoint comes in (Test data) it passes through all the models (weak learners) and the class which gets the highest vote is the output for our test data.

## 1.3 Improvements to Basic Gradient Boosting<a class="anchor" id="1.3"></a>
### Gradient boosting is a greedy algorithm and can overfit a training dataset quickly.

It can benefit from regularization methods that penalize various parts of the algorithm and generally improve the performance of the algorithm by reducing overfitting.

In this this section we will look at 4 enhancements to basic gradient boosting:

+ Tree Constraints
+ Shrinkage
+ Random sampling
+ Penalized Learning

1. Tree Constraints
It is important that the weak learners have skill but remain weak.

There are a number of ways that the trees can be constrained.

A good general heuristic is that the more constrained tree creation is, the more trees you will need in the model, and the reverse, where less constrained individual trees, the fewer trees that will be required.

Below are some constraints that can be imposed on the construction of decision trees:

Number of trees, generally adding more trees to the model can be very slow to overfit. The advice is to keep adding trees until no further improvement is observed.
Tree depth, deeper trees are more complex trees and shorter trees are preferred. Generally, better results are seen with 4-8 levels.
Number of nodes or number of leaves, like depth, this can constrain the size of the tree, but is not constrained to a symmetrical structure if other constraints are used.
Number of observations per split imposes a minimum constraint on the amount of training data at a training node before a split can be considered
Minimim improvement to loss is a constraint on the improvement of any split added to a tree.

2. Weighted Updates
The predictions of each tree are added together sequentially.

The contribution of each tree to this sum can be weighted to slow down the learning by the algorithm. This weighting is called a shrinkage or a learning rate.

3. Stochastic Gradient Boosting
A big insight into bagging ensembles and random forest was allowing trees to be greedily created from subsamples of the training dataset.

This same benefit can be used to reduce the correlation between the trees in the sequence in gradient boosting models.

This variation of boosting is called stochastic gradient boosting.A few variants of stochastic boosting that can be used:

+ Subsample rows before creating each tree.
+ Subsample columns before creating each tree
+ Subsample columns before considering each split.

4. Penalized Gradient Boosting
Additional constraints can be imposed on the parameterized trees in addition to their structure.

+ L1 regularization of weights.
+ L2 regularization of weights.

#### Gradient boosting is a greedy algorithm and can overfit a training dataset quickly. So regularization methods are used to improve the performance of the algorithm by reducing overfitting.

+ **Subsampling:** This is the simplest form of regularization method introduced for GBM’s. This improves the generalization properties of the model and reduces the computation efforts. Subsampling introduces randomness into the fitting procedure. At each learning iteration, only a random part of the training data is used to fit a consecutive base-learner. The training data is sampled without replacement.
+ **Shrinkage:** Shrinkage is commonly used in ridge regression where it shrinks regression coefficients to zero and, thus, reduces the impact of potentially unstable regression coefficients. In GBM’s, shrinkage is used for reducing the impact of each additionally fitted base-learner. It reduces the size of incremental steps and thus penalizes the importance of each consecutive iteration. The intuition behind this technique is that it is better to improve a model by taking many small steps than by taking fewer large steps. If one of the boosting iterations turns out to be erroneous, its negative impact can be corrected easily in subsequent steps.
+ **Early Stopping:** One important practical consideration that can be derived from Decision Tree is early stopping or tree pruning. This means that if the ensemble was trimmed by the number of trees, corresponding to the validation set minima on the error curve, the overfitting would be circumvented at the minimal accuracy expense. Another observation is that the optimal number of boosts, at which the early stopping is considered, varies concerning the shrinkage parameter λ. Therefore, a trade-off between the number of boosts and λ should be considered.

## 1.4 Summary:<a class="anchor" id="1.4"></a>
Gradient boosting involves three elements:

+ 1. A loss function to be optimized.
+ 2. A weak learner to make predictions.
+ 3. An additive model to add weak learners to minimize the loss function.

1. Loss Function
The loss function used depends on the type of problem being solved.

It must be differentiable, but many standard loss functions are supported and you can define your own.

For example, regression may use a squared error and classification may use logarithmic loss.

2. Weak Learner
Decision trees are used as the weak learner in gradient boosting.

Specifically regression trees are used that output real values for splits and whose output can be added together, allowing subsequent models outputs to be added and “correct” the residuals in the predictions.

Trees are constructed in a greedy manner, choosing the best split points based on purity scores like Gini or to minimize the loss.

Initially, such as in the case of AdaBoost, very short decision trees were used that only had a single split, called a decision stump. Larger trees can be used generally with 4-to-8 levels.

It is common to constrain the weak learners in specific ways, such as a maximum number of layers, nodes, splits or leaf nodes.

This is to ensure that the learners remain weak, but can still be constructed in a greedy manner.

3. Additive Model
Trees are added one at a time, and existing trees in the model are not changed.

A gradient descent procedure is used to minimize the loss when adding trees.

Traditionally, gradient descent is used to minimize a set of parameters, such as the coefficients in a regression equation or weights in a neural network. After calculating error or loss, the weights are updated to minimize that error.

Instead of parameters, we have weak learner sub-models or more specifically decision trees. After calculating the loss, to perform the gradient descent procedure, we must add a tree to the model that reduces the loss (i.e. follow the gradient). We do this by parameterizing the tree, then modify the parameters of the tree and move in the right direction by (reducing the residual loss.

The output for the new tree is then added to the output of the existing sequence of trees in an effort to correct or improve the final output of the model.

A fixed number of trees are added or training stops once loss reaches an acceptable level or no longer improves on an external validation dataset.

[Table of Content](#0.1)
## 2. Extreme Gradient Boosting Machine (XGBM)<a class="anchor" id="2"></a>

![image](https://user-images.githubusercontent.com/99672298/180611820-2137c89b-1484-418d-bde8-7818814751a2.png)

XGBoost is an extension to gradient boosted decision trees (GBM) and specially designed to improve speed and performance. In fact, XGBoost is simply an improvised version of the GBM algorithm! The working procedure of XGBoost is the same as GBM. `Regularized Learning`, `Gradient Tree Boosting` and `Shrinkage and Column Subsampling`. The trees in XGBoost are built sequentially, trying to correct the errors of the previous trees.

#### XGBoost Features
+ **Regularized Learning:** The regularization term helps to smooth the final learned weights to avoid over-fitting. The regularized objective will tend to select a model employing simple and predictive functions.
Gradient Tree Boosting: The tree ensemble model cannot be optimized using traditional optimization methods in Euclidean space. Instead, the model is trained in an additive manner.
+ **Shrinkage and Column Subsampling:** Besides the regularized objective, two additional techniques are used to further prevent overfitting. The first technique is shrinkage introduced by Friedman. Shrinkage scales newly added weights by a factor η after each step of tree boosting. Similar to a learning rate in stochastic optimization, shrinkage reduces the influence of each tree and leaves space for future trees to improve the model.
+ The second technique is the column (feature) subsampling. This technique is used in Random Forest. Column sub-sampling prevents over-fitting even more so than the traditional row sub-sampling. The usage of column sub-samples also speeds up computations of the parallel algorithm.

#### But there are certain features that make XGBoost slightly better than GBM:

+ One of the most important points is that XGBM implements parallel preprocessing (at the node level) which makes it faster than GBM and that means using Parallel learning to split up the dataset so that multiple computers can work on it at the same time.
+ XGBoost also includes a variety of regularization techniques that reduce overfitting and improve overall performance. You can select the regularization technique by setting the hyperparameters of the XGBoost algorithm
+ Additionally, if you are using the XGBM algorithm, you don’t have to worry about imputing missing values in your dataset. The XGBM model can handle the missing values on its own. During the training process, the model learns whether missing values should be in the right or left node.

#### In other words, the first three parts give us a conceptual idea of How XGBoost is fit to training data and how it makes predictions
and the other parts we are going to discuss are going to describe optimization techniques for large datasets
![22 07 2022_16 05 44_REC](https://user-images.githubusercontent.com/99672298/180612232-b6f1e813-5f3e-4632-b3ad-c055d0e0b137.png)

#### XGBM Optimizations:
+ **Exact Greedy Algorithm:** The main problem in tree learning is to find the best split. This algorithm enumerates all the possible splits on all the features. It is computationally demanding to enumerate all the possible splits for continuous features.
+ **Approximate Algorithm:** The exact greedy algorithm is very powerful since it enumerates overall possible splitting points greedily. However, it is impossible to efficiently do so when the data does not fit entirely into memory. Approximate Algorithm proposes candidate splitting points according to percentiles of feature distribution. The algorithm then maps the continuous features into buckets split by these candidate points, aggregates the statistics, and finds the best solution among proposals based on the aggregated statistics. So when we have huge training dataset, XGBoost uses an Approximate Greedy Algorithm.
++ **Weighted Quantile Sketch:** One important step in the approximate algorithm is to propose candidate split points. XGBoost has a distributed weighted quantile sketch algorithm to effectively handle weighted data.
Sparsity-aware Split Finding: In many real-world problems, it is quite common for the input x to be sparse. There are multiple possible causes for sparsity:
Presence of missing values in the data
Frequent zero entries in the statistics
Artifacts of feature engineering such as one-hot encoding
It is important to make the algorithm aware of the sparsity pattern in the data. XGBoost handles all sparsity patterns in a unified way.

#### System Features
The library provides a system for use in a range of computing environments, not least:

+ **Parallelization** of tree construction using all of your CPU cores during training.
+ **Distributed Computing** for training very large models using a cluster of machines.
+ **Out-of-Core Computing** for very large datasets that don’t fit into memory.
+ **Cache Optimization** of data structures and algorithm to make the best use of hardware.

#### Algorithm Features
The implementation of the algorithm was engineered for the efficiency of computing time and memory resources. A design goal was to make the best use of available resources to train the model. Some key algorithm implementation features include:

+ **Sparse Aware implementation** with automatic handling of missing data values.
+ **Block Structure** to support the parallelization of tree construction.
+ **Continued Training** so that you can further boost an already fitted model on new data.



[Table of Content](#0.1)
## Maths Intuition
### 1.5 Understand Gradient Boosting Algorithm with example (Regression)<a class="anchor" id="1.5"></a>
Let’s understand the intuition behind Gradient boosting with the help of an example. Here our target column is continuous hence we will use Gradient Boosting Regressor.

Following is a sample from a random dataset where we have to predict the car price based on various features. The target column is price and other features are independent features.

![image](https://user-images.githubusercontent.com/99672298/180592447-05d51d72-bd76-40b2-850f-da745d8e0e75.png)\
_______________________________________________________________________________________________________________________________________________________________
#### Step -1 The first step in gradient boosting is to build a base model to predict the observations in the training dataset. For simplicity we take an average of the target column and assume that to be the predicted value as shown below:
_______________________________________________________________________________________________________________________________________________________________
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
_______________________________________________________________________________________________________________________________________________________________
#### Step-2 The next step is to calculate the pseudo residuals which are (observed value – predicted value)
_______________________________________________________________________________________________________________________________________________________________

![20 07 2022_20 25 10_REC](https://user-images.githubusercontent.com/99672298/180602144-1cba6543-31a0-437f-b893-847962ac1744.png)

Again the question comes why only observed – predicted? Everything is mathematically proved, let’s from where did this formula come from. This step can be written as:

![image](https://user-images.githubusercontent.com/99672298/180592566-be077eb8-3843-4735-bf10-4269b26fd5e0.png)

Here F(xi) is the previous model and m is the number of DT made.

The predicted value here is the prediction made by the previous model. In our example the prediction made by the previous model (initial base model prediction) is 14500, to calculate the residuals our formula becomes:

![image](https://user-images.githubusercontent.com/99672298/180592632-e80350f1-b5f7-4239-99d3-1f69af26087d.png)
![image](https://user-images.githubusercontent.com/99672298/180592557-6acc1beb-8907-4353-af6f-8ddb627f0055.png)

In the next step, we will build a model on these pseudo residuals and make predictions. Why do we do this? Because we want to minimize these residuals and minimizing the residuals will eventually improve our model accuracy and prediction power. So, using the Residual as target and the original feature Cylinder number, cylinder height, and Engine location we will generate new predictions. Note that the predictions, in this case, will be the error values, not the predicted car price values since our target column is an error now.

Let’s say hm(x) is our DT made on these residuals.
_______________________________________________________________________________________________________________________________________________________________
#### Step- 3 In this step we find the output values for each leaf of our decision tree. That means there might be a case where 1 leaf gets more than 1 residual, hence we need to find the final output of all the leaves. TO find the output we can simply take the average of all the numbers in a leaf, doesn’t matter if there is only 1 number or more than 1.\
_______________________________________________________________________________________________________________________________________________________________
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
_______________________________________________________________________________________________________________________________________________________________
#### Step-4 This is finally the last step where we have to update the predictions of the previous model. It can be updated as:
_______________________________________________________________________________________________________________________________________________________________

![image](https://user-images.githubusercontent.com/99672298/180592838-6d150e8f-9cd1-4b1c-a9b7-defced68e81b.png)

where m is the number of decision trees made.

Since we have just started building our model so our m=1. Now to make a new DT our new predictions will be:

![image](https://user-images.githubusercontent.com/99672298/180592844-6b95fe59-f048-49f7-8ff9-3bb64b6fa3d9.png)

Here Fm-1(x) is the prediction of the base model (previous prediction) since F1-1=0 , F0 is our base model hence the previous prediction is 14500.

nu is the learning rate that is usually selected between 0-1. It reduces the effect each tree has on the final prediction, and this improves accuracy in the long run. Let’s take nu=0.1 in this example.

Hm(x) is the recent DT made on the residuals.

Let’s calculate the new prediction now:

![image](https://user-images.githubusercontent.com/99672298/180592852-56692fcf-636a-41d1-aaee-0f16474df415.png)

[Table of Content](#0.1)
## Maths Intuition
### 1.6 Gradient Boosting Classifier<a class="anchor" id="1.6"></a>
What is Gradient Boosting Classifier?
A gradient boosting classifier is used when the target column is binary. All the steps explained in the Gradient boosting regressor are used here, the only difference is we change the loss function. Earlier we used Mean squared error when the target column was continuous but this time, we will use log-likelihood as our loss function.

![image](https://user-images.githubusercontent.com/99672298/180600358-32739748-9481-4bfd-bba3-7e58dc86f6eb.png)

Let’s see how this loss function works,
The first step is creating an initial constant prediction value F₀. L is the loss function and we are using log loss (or more generally called cross-entropy loss) for it.
_______________________________________________________________________________________________________________________________________________________________
### Step 1
_______________________________________________________________________________________________________________________________________________________________
![image](https://user-images.githubusercontent.com/99672298/180600390-f0be5a77-4591-4c8b-8a1e-2f31009b59ad.png)

![image](https://user-images.githubusercontent.com/99672298/180598331-ffc05535-2648-43c4-bc9b-806142c2a406.png)

yᵢ is our classification target and it is either 0 or 1. p is the predicted probability of class 1. You might see L taking different values depending on the target class yᵢ.

![image](https://user-images.githubusercontent.com/99672298/180598339-1f18ea4e-3c6a-4f91-9728-cad35b91faae.png)

As −log(x) is the decreasing function of x, the better the prediction (i.e. increasing p for yᵢ=1), the smaller loss we will have.

argmin means we are searching for the value γ (gamma) that minimizes ΣL(yᵢ,γ). While it is more straightforward to assume γ is the predicted probability p, we assume γ is log-odds as it makes all the following computations easier. For those who forgot the log-odds definition , it is defined as log(odds) = log(p/(1-p)).

To be able to solve the argmin problem in terms of log-odds, we are transforming the loss function into the function of log-odds.

Our first step in the gradient boosting algorithm was to initialize the model with some constant value, there we used the average of the target column but here we’ll use log(odds) to get that constant value. The question comes why log(odds)?

When we differentiate this loss function, we will get a function of log(odds) and then we need to find a value of log(odds) for which the loss function is minimum.

Confused right? Okay let’s see how it works:

Let’s first transform this loss function so that it is a function of log(odds), I’ll tell you later why we did this transformation.

![image](https://user-images.githubusercontent.com/99672298/180598182-fa3f6c4b-05ba-40d8-b0d4-2ccce4dae77b.png)

Now we might want to replace p in the above equation with something that is expressed in terms of log-odds. By transforming the log-odds expression shown earlier, p can be represented by log-odds:

![image](https://user-images.githubusercontent.com/99672298/180598541-74d8e309-9fdc-4ebb-9849-716cda4754f2.png)

Then, we are substituting this value for p in the previous L equation and simplying it.

![image](https://user-images.githubusercontent.com/99672298/180599089-9e0ddfc0-8f5b-4861-b1be-8bd58d736d01.png)

Now this is our loss function, and we need to minimize it, for this, we take the derivative of this w.r.t to log(odds) and then put it equal to 0,

![image](https://user-images.githubusercontent.com/99672298/180600153-6b966f24-22b0-4eb3-89bb-9af03278c1bf.png)\

In the equations above, we replaced the fraction containing log-odds with p to simplify the equation. Next, we are setting ∂ΣL/∂log(odds) equal to 0 and solving it for p.

![image](https://user-images.githubusercontent.com/99672298/180600167-184bf77d-82b5-4fa2-a56b-1b45ed99f8c0.png)

In this binary classification problem, y is either 0 or 1. So, the mean of y is actually the proportion of class 1. You might now see why we used p = mean(y) for our initial prediction.

As γ is log-odds instead of probability p, we are converting it into log-odds.

![image](https://user-images.githubusercontent.com/99672298/180600342-5b65fb23-a0de-43cf-9786-2bffe3cd5232.png)
_______________________________________________________________________________________________________________________________________________________________
### Step2
_______________________________________________________________________________________________________________________________________________________________

![image](https://user-images.githubusercontent.com/99672298/180600786-92aa4e79-b132-4408-9caa-125175bb7051.png)

The whole step2 processes from 2–1 to 2–4 are iterated M times. M denotes the number of trees we are creating and the small m represents the index of each tree.
_______________________________________________________________________________________________________________________________________________________________
#### Step2-1
_______________________________________________________________________________________________________________________________________________________________


![image](https://user-images.githubusercontent.com/99672298/180600692-5b95f055-62d3-49d9-be58-220ba93165d6.png)

We are calculating residuals rᵢ𝑚 by taking a derivative of the loss function with respect to the previous prediction F𝑚-₁ and multiplying it by −1. As you can see in the subscript index, rᵢ𝑚 is computed for each single sample i. Some of you might be wondering why we are calling this rᵢ𝑚 residuals. This value is actually negative gradient that gives us the directions (+/−) and the magnitude in which the loss function can be minimized. You will see why we are calling it residuals shortly. By the way, this technique where you use a gradient to minimize the loss on your model is very similar to gradient descent technique which is typically used to optimize neural networks. (In fact, they are slightly different from each other.

Let’s compute the residuals here. F𝑚-₁ in the equation means the prediction from the previous step. In this first iteration, it is F₀. As in the previous step, we are taking a derivative of L with respect to log-odds instead of p since our prediction F𝑚 is log-odds. Below we are using L expressed by log-odds which we got in the previous step.

![image](https://user-images.githubusercontent.com/99672298/180600756-026564dd-fe2c-4548-9885-5e4d21b90286.png)

In the previous step, we also got this equation:

![image](https://user-images.githubusercontent.com/99672298/180600760-bf137c5f-0e24-44c2-9dd7-eae2f8dd77d5.png)

So, we can replace the second term in rᵢ𝑚 equation with p.

![image](https://user-images.githubusercontent.com/99672298/180600767-9e79e9b5-42fc-49af-9a75-f45dc67b9417.png)

You might now see why we call r residuals. This also gives us interesting insight that the negative gradient that provides us the direction and the magnitude to which the loss is minimized is actually just residuals.

-----------------------------------------------------------------------------OR-----------------------------------------------------------------------------

![image](https://user-images.githubusercontent.com/99672298/180598190-6bf2c163-ad33-43ce-a07d-77e556163a21.png)

Here y are the observed values\
You must be wondering that why did we transform the loss function into the function of log(odds). Actually, sometimes it is easy to use the function of log(odds), and sometimes it’s easy to use the function of predicted probability “p”.

It is not compulsory to transform the loss function, we did this just to have easy calculations.

Hence the minimum value of this loss function will be our first prediction (base model prediction)

Now in the Gradient boosting regressor our next step was to calculate the pseudo residuals where we multiplied the derivative of the loss function with -1. We will do the same but now the loss function is different, and we are dealing with the probability of an outcome now.

![image](https://user-images.githubusercontent.com/99672298/180598203-60423cee-c552-44f7-ae93-a1c057db4adb.png)

After finding the residuals we can build a decision tree with all independent variables and target variables as “Residuals”.

Now when we have our first decision tree, we find the final output of the leaves because there might be a case where a leaf gets more than 1 residuals, so we need to calculate the final output value. 

![image](https://user-images.githubusercontent.com/99672298/180598211-b3ab87e1-f5b0-40af-80e4-4959b5a0f046.png)

Finally, we are ready to get new predictions by adding our base model with the new tree we made on residuals.
_______________________________________________________________________________________________________________________________________________________________
[Table of Content](#0.1)

