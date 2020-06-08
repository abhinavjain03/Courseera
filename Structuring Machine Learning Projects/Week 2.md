# Week 2
## Error Analysis
* When trying to fix a problem , analyse how much will that improve the performance of the system.
* * Evaluate several ideas in parallel - Calculate % of errors being contributed by the individual problems (by checking a set of mislabelled examples). That can decide which problem to focus on.

## Cleaning up incorrectly labelled data
* Incorrect label in Training set - DL algorithms are robust to random errors in training set if the error is random and very low %age. -> Not robust to systematic errors
* Incorrect label in Dev/test set - Check the % of incorrect labels while analysing problems and if it is a relatively large number as compared to other problems, fix the labels.
* Notes for corrting labels:
  * Apply same process of label correction to both dev and test set

## Build your first system quickly and iterate
* Set up a dev/test set and metric
* Build initial system quickly
* Bias-Variance and Error analysis to priorize problems

## Mismatched training and dev/test data
* Dev/test should contain data depicting your problem
* Can use multiple datasets in training

## Error analysis with mismatched data distributions
* If train set and dev set errors have a huge gap, it can be due to 2 reasons - *different distributions of training and dev sets* or *the variance problem*
  * Can create a **training-dev** set, a subset of the training set which have the same distribution as training set but is not used in training
  ![Error analysis - Data mismatch](imgs/error-analysis-mismatch-data.png)

### Error Analysis
![Error analysis - Data mismatch](imgs/error-analysis-mismatch-data-2.png)

**Note**: The training-dev set error can be more than the dev and test set errors if the dev/test sets are easier.

### Addressing Data Mismatch
* Understand differences between train and dev sets - like traing set is clean while dev set is noisy
* Make training data more similar to dev/test set
* Artificial data synthesis - Reverberation or Noise augmentation to clean audio

## Transfer Learning
* Apply knowledge from one task and apply to other task. Adapt/transfer.
* Eg. (pretraining) train model on image recognition, remove output layer, add new output layers for another task (radiology diagnosis) and train again using radiology data (finetuning)
* Transfer learning from A to B makes sense when -
  * Input from A and B are same
  * data for task A is much more than task B
  * Low level features for both task A and task B are similar

## Multitask Learning
* In MTL, one NN is trained simultaneously on multiple tasks
*  Single input, multiple outputs, one for each task.
* MTL loss - 
  ![MTL](imgs/mtl.png)
  **Note**: 4 is the number of tasks here.
* Initial layers can share the initial layer features.
* MTL makes sense when -
  * when multiple tasls can share lower level features
  * amount of data for each task is similar - usually

## End-to-End Deep Learning
* Replace multiple stages of a problem with a single Deep Network
* Eg. Speech Recognition -> Going from features(MFCC), to phones, to words, to sentences -> to a single network
* Needs a whole lot more data

### When to use E2E or not
* Pros -
  * Let the data speak - rather than humans feeding specific features
  * Less hand desigining components
* Cons - 
  * Requires lots of data
  * Hand designed components are excluded - they are created using human expertise



