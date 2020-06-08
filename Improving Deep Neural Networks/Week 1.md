# Week 1 
* Applied ML is a highly iterative process.
* Intuitions from one domain do not translate to other domains - depends on data, CPU GPU config.

## Splitting data into Train/Dev/Test sets
* Carve some portion as training set, some to be dev set, and some to be the test set.
* Train on train set, evaluate on dev set, repeat fro multiple times and evaluate the final model on the test set.
* Traditionally, 70-30 or 60-20-20 or splits as data is small.
* In deep learning era, test and dev set size have reduced. Eg. if have 100,000 points, have 30% dev set is unreasonable. More like, 98-1-1 splits

### Mismatched train-test distributions
* the distribution of train and test set should be same. 
* Eg. Train set - cat pictures from web (high quality) Test - cat pictures from app (low quality) will have different distributions.

## Bias-Variance
* Less tradeoff in the Deep learning era.
* In 2-d, one can plot and see bias-variance relationship.
![Bias-variance](imgs/b-v.png)
* For NNs, we can analyse bias and variance using Train-Dev Errors.
![Bias-variance NN](imgs/b-v-nn.png)
**Note**: In the second column, it is high bias on the assumtion that Human Error(Bayes error) on this task is very low, say 0%.

**Note**: This is under the assumption that Train and Dev sets are from the same distribution

| Condition | Analysis |
| :-: | :- |
| High Bias High Variance | Model is not powerful enough, and it is overfitting on the data |
| High bias Low Variance | Model is not powerful enough, but no overfitting |
| Low Bias High Variance | Model is powerful enough, but overfitting on the train data |
| Low Bias Low Variance | Model is powerful enough and no overfitting |

## Basic Recipe for ML
* Start training with some model.
* Check for *High Bias* i.e. training error, if its large -> try bigger network, more layers, more nodes, training longer
* Check for *High Variance* i.e dev set error, if its large -> get more data, regularization, more appropriate NN architecture (NAS)

## Bias Variance tradeoff
* In pre Deep learning era, "Bias Variance tradeoff" was a problem. We had tools just to reduce bias or just to reduce variance, but those tend to effect the other.
* In Deep learning, 
  * Adding more layers -> reduces bias alomost always
  * Adding more data -> reduces variance almost always

  without effecting the other

## Regularization
Solutions for overfitting or High Variance
* apply Regularization
* Get more training data(Not possible always)

* In LR, apply regularization on the parameters of the model.
![Regularization - LR](imgs/regularize-LR.png)
   * **L2 Regularization** - Euclidean norm of the weights scaled with *lambda/2m*, regularization parameter is added to the loss *J*. 
    ```python
                 (lambda/2m)*np.dot(w.T,w)
    ```
   * **L1 Regularization** - L1 norm of the weights scaled with *lambda/2m*, regularization parameter is added to the loss *J*. Using this, w will end up being a sparse matrix (meaning w will have lots of 0's). 
   ```python
               (lambda/2m)*np.sum(np.abs(w))
   ```
**Note:** *lambda* is a hyper parameter. 
**Note:** Usually omit *b*, as it is very less number as compared to *w*.

* Regularization in Neural Networks
   * Add **Frobenius norm** to the loss function of the NN.
![Regularize - NN](imgs/regularize-NN.png)

   * **Implementation of L2 Regularization** - During derivative computation of w, dw, add the gradient of the regularization term too.

      So, `dw_l = (from backprop) + (lambda/m)*w_l`

**Note:** L2 Regularization is also called **weight-decay**. As we are making it a little smaller as we subtract the regularization term from w.

### Why Regularization reduces overfitting?
* On penalizing weights, we try to reduce the effect of multiple nodes on the outcome of the NN. That can reduce overfitting.
* Eg: intuition, suppose we have a model with High Variance(overfitting), we have a loss with L2 regularization constant. If this constant is very large, it will push the model weights towards 0, so this new model can be taken as a less complex model, towards the High Bias side, hence reducing overfitting. 
* Eg: intuition, suppose we have a NN with tanh non linearity. If we apply Regularization constant lambda, and its large, the weights will be small and the output of the tanh function will be limited to the linear part. So the NN will move towards being more linear. Hence reducing overfitting.

### Dropout
* With some probability, drop the nodes of each layer of the NN.
* Makes the network simpler on using regularization

#### Implementing Dropout (Inverted dropout) - 
Let's add dropout layer after layer3.
* Output of layer 3 -> *a3*, keep_probs=0.8 (let's say)
* `d3 = np.random.randn(a3.shape[0], a3.shape[1]) < keep_probs`.   
  **Note**: This makes d3 a boolean array but on elementwise multiplication, this will be treated as 0 or 1.
* `a3 = np.multiply(a3, d3)` -> elementwise multiplication
* `s3 /= keep_probs` -> for scaling such that the expected value of z4 doesnt change. (`z4 = np.dot(w4,a3) + b4` and a3 will be reduced bt 20%, so we scale the a3 by 0.8) - This is **inverted dropout**

During test time, we dont use dropout. And we dont need any scaling as we have already used inverted dropout (Scaling during training).

### Why Dropout works?
Intuition - Can't rely on any one feature, so have to spread out weights. 
* As dropout randomly drops out any node, so the output node will try to spread the weights onto other nodes (Shrink weights - adaptive form of L2 regularization but penalty on each weight is different which depends on the activation)

**Note**: keep_prob can be different for different layers.
 
**Note**: Dropout can change the loss plot. 

### Other Regularization Methods
* **Data Augmentation** - Like flipping images, random crops of image, rotations and distortions to characters, in case of OCR.
* **Early Stopping** - 
![Early Stopping](imgs/early-stopping.png)
Just stop earlier.
*Note:*Drawback is we are removing orthogonalizing between tasks. Now not overfitting and optimize loss fuction are dependent.

## Normalizing Inputs
* Subtract mean from feature (making it 0 mean)
* Dividing with standard deviation (Normalize variance and making it 1)
   
   **Note**: Normalize test set using the same mean and variance.
### Why normalization?
* Brings inputs to the same scale. So weights will also come to the same scale. The optimization function will be symmetrical and descent will be easier.
![Normalize](imgs/normalize.png)

## Vanishing and Exploding Gradients
* While training Deep networks, gradients can become very very big or very very small, making training difficult.
* If weights are larger, and as forward pass requires multiplication of weights number of layers times, so hat{y} will be w^(num_layers).
* Similar with if the weights are smaller, values will be very small.
* And so will the gradients will increase or decrease exponentially.

## Weight Initialization of DNNs
* Partial solution of Exploding or Vanishing gradients

The initialization should depend on the input of that node. More number of inputs, small weights. 
  * If using *ReLU* activation, var=2/n. 
    ```python
        w[l] = np.random.randn(shape)*np.sqrt(2/n[l-1])
    ```
  * If using *tanh* activation, var=1/n. 
    ```python
        w[l] = np.random.randn(shape)*np.sqrt(1/n[l-1]) 
    ```
  * Or 
    ```python
        w[l] = np.random.randn(shape)*np.sqrt(1/(n[l-1]+n[l]))
    ```
## Numerical Approximation of Gradients
![Numerical Approximation](imgs/numerical-approx-gradient.png)

**Note**: Using 2 sided difference gives us better confidence over the gradient. One sided error->O(eps), 2 sided error->O(eps^2) and eps is very small.


## Gradient Checking
* Take all the parameters and reshape them into 1-d vector. 
* Concatenate all the vectors to form 1 giant vector. 
* Do the same for all the gradients also.

![Gradient Check](imgs/grad-check.png)
* The euclidean distance by norm value should be as close to epsilon as possible for everything to be perfect.

### Implementation 
* Dont use in training, only to debug, as looping will take a lot of time.
* If grad check fails, check for individual components. Those values which are very far from the original, then there is the problem.
* If applying regularization, use that in grad.
* Doesnt work with Dropout. Use grad_check with keep_prob 1.0, check grad and then turn on dropout.
* Run at random initialization.