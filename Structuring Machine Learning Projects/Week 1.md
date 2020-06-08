# Week 1
## Orthogonalization
* Orthogonal means 90 degress of each other
* Separation of concerns - separate solutions for separate problems
* Assumptions in ML:
  * Fits training set well on cost function
    * Bigger network
    * Better optimization algo
  * Fits dev set well on cost function
    * Regularization
  * Fits test set well on cost funtion
    * Bigger dev set
  * Performs well in real world
    * Change dev set or cost function 

**Note**: Don't use early stopping, as it effects both performance on training set and performance on dev set.

## Single number evaluation metric
* Suppose have a classifier, we use Precision and Recall as 2 measures. 
* If we have two models, it hard selecting the better out of them as there are two evaluation metrics.
* Better to use F1 score, as it combines both Precision and Recall, as a single number evaluation metric.

### Satisficing and Optimizing metrics
* Supoose you have multiple metrics, which can't be linearly combined.
* You can choose them as satisficing and optimizing metrics.
* Eg. A classifier, two metrics, accuracy and running time.
  * As accuracy is what we will try to maximize, it will be the optimizing metric -> as you want this as better as bpossible.
  * subject to running time to be less than some value, so running time will be our satisficing metric -> as it will work to be less than some threshold
 * Eg. wake up devices, two metrics, accuracy and # false positives per 24 hrs.
   * Accuracy can be optimizing metric
   * \# false positives less than 4 per day will be the satisficing metric


 **Note**: N metrics, so 1 metric as optimizing, and N-1 as satisficing metrics.

## Train/Dev/Test Distributions
* Train on train set, evaluate on dev set to make it better and better. Final performance on test set.
* Choose the dev set and the test set to reflect the data you expect to get in future.
* Dev and Test set should be from the same distribution.

### Size of the dev/test sets
* Old way of splitting data -> 70/30 or 60/20/20 (when dataset were smaller)
* This split no longer applies as dataset sizes are very large.
* Size of the test set to be big enough to give high confidence in the performance

### When to change dev/test sets and metrics
* When the evaluation metric doesnt correctly rank the algorithms we try, then change the metric or dev set. 

## Comparing to human-level performance
* Deep learning has made the performance of lots of algorithms better, so its feasible to compare them with human level performance.
* Comparing with human level performance can help to increase the performance of the ML algo.
  * Get more labeled data from humans
  * Bias-Variance analysis depends on human level performance
* Bayes optimal error is the error which cant be passed.
![Bayes Optimal Error](imgs/bayes-optimal-error.png)
* Human level performance is really close to Bayes optimal error.

## Avoidable Bias
* We want out algorithm to learn teh training set well but not too well.

![Bias Variance](imgs/bias-variance.png)
* Bias Variance analysis depends on human level performance.

* **Avoidable Bias** - Difference between the (human level performance)/(bayes error) and the training set error of the algorithm.
  * If this is high, there is room for reducing this.
* Diff between the training error and the dev set error is the variance.

## Understanding Human level performance
* using human level error as a proxy for bayes error
* Choose human level performance estimate carefully.
* It is harder to improve when you are closer to human level error
* Bias variance analysis is done keeping in mind the bayes error.

### Surpassing human level performance
* Egs where ML surpasses human level performance - 
  * Online advertising
  * Product Recommendations
  * Logistics
  * Loan approvals
* This is learning from structured data, not natural perception problems like speech recognition, or computer vision

## Improving your model performance
### Two fundamental assumptions
* Fit on training data really well (Less Avoidable bias). If more avoidable bias:
    * Train bigger model
    * Train longer
    * Better optimization algos
    * Better NN architecture/Hyperparameter search
* Training set performance generalizes well on dev set (Variance is not too bad). If variance is large: 
  * Regularization -> L2, dropout, data augmentation
  * NN architecture/hyperparameter search