```python
!pip install -q scikit-learn==1.0.1
from sklearn import metrics
import numpy as np
```

### Definition

!!! success "Definition"
    Formally, if $\hat{y}^{(i)}$ is the predicted value of the i-th sample and the ground truth is $y^{(i)}$, then accuracy can be defined as the fraction of predictions that our classifier/hypothesis/model predicted correctly, over the total number of samples in question.

$$
\begin{aligned}
\text{accuracy}(\hat{y}^{(i)}, y^{(i)}) &= \dfrac{1}{\text{num_samples}}\sum_{i=1}^{\text{num_samples}}\mathrm{1}(y^{(i)}\hat{y}^{(i)}) \\ 
                                        &= \dfrac{\text{Number of correctly classified cases}}{\text{Number for all cases}}
\end{aligned}
$$

where $\mathrm{1}(x)$ is the [indicator function](https://en.wikipedia.org/wiki/Indicator_function).


!!! note
    Accuracy is a simple enough metric such that the definition for both binary and multiclass classification is the same.

### When to use Accuracy as a metric

Classes are well balanced: Accuracy is a valid choice of evaluation for classification problems which are well balanced and not skewed or no class imbalance. Typically, one should plot EDA and see the classes - if they are roughly equal, then `accuracy` can be used. However, accuracy should not be the only metric to look at in a classification problem.


### When NOT to use Accuracy as a metric

???+ danger
    Consider an imbalanced set, where the training data set has 100 patients (data points), and the ground truth is 90 patients are of class = 0, which means that these patients do not have cancer, whereas the remaining 10 patients are in class 1, where they do have cancer. This is an example of class imbalance where the ratio of class 1 to class 0 is $1:9$.

    Next, we consider **a baseline (almost trivial) classifier**:

    ```python
    def baselineModel(patient_data):
            training...
        return benign
    ```

    where we predict the patient's class as the most frequent class. Meaning, the most frequent class in this question is the class = 0, where patients do not have cancer, so we just assign this class to everyone in this set. By doing this, we will inevitably achieve a **in-sample** accuracy rate of $\frac{90}{100} = 90\%$. But unfortunately, this supposedly high accuracy value is completely useless, because this classifier did not label any of the cancer patients correctly.

    The consequence can be serious, assuming the test set has the same distribution as our training set, where if we have a test set of 1000 patients, there are 900 negative and 100 positive. Our model just literally predict every one of them as benign, yielding a $90\%$ **out-of-sample** accuracy.

    What did we conclude? Well, for one, our `accuracy` can be 90% high and looks good to the laymen, but it failed to predict the most important class of people.



### Implementation of Accuracy


```python
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates accuracy score of a prediction.

    Can be used for both binary and multiclass classification.

    Args:
        y_true (np.ndarray): the correct labels, shape (n_samples, )
        y_pred (np.ndarray): the predicted labels, shape (n_samples, )

    Returns:
        accuracy_score (float): the accuracy score
    """

    accuracy_count = 0  # numerator
    num_samples = len(y_true)  # denominator

    for y_t, y_p in zip(y_true, y_pred):
        if y_t == y_p:
            accuracy_count += 1

    accuracy_score = accuracy_count / num_samples
    return accuracy_score
```


```python
# Binary Classification
y_true = np.asarray([1,1,0,1,0,0])
y_pred = np.asarray([1,1,1,0,0,0])
print(f"hn accuracy: {accuracy(y_true, y_pred)}")
print(f"sklearn accuracy: {metrics.accuracy_score(y_true, y_pred, normalize=True)}")

# Multiclass Classification
y_pred = np.asarray([0, 2, 1, 3])
y_true = np.asarray([0, 1, 2, 3])
print(f"hn accuracy: {accuracy(y_true, y_pred)}")
print(f"sklearn accuracy: {metrics.accuracy_score(y_true, y_pred, normalize=True)}")
```

    hn accuracy: 0.6666666666666666
    sklearn accuracy: 0.6666666666666666
    hn accuracy: 0.5
    sklearn accuracy: 0.5
    
