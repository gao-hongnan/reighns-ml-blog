This notebook attempts to describe the commonly used Classification Metrics in details; you will find the pros and cons of each metric, alongside with examples to illustrate. I will refer to scikit-learn's official website for reference, specifically, the [section on Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html). I may quote some definitions verbatim, if necessary.

We will be going through three major types of classification problems.

- Binary Classification
- Multiclass Classification
- Multilabel Classification

---

## Motivation

Unlike Kaggle competitions, metrics have to be carefully chosen in business. A wrong metric may lead to disastrous outcomes. We open with a classic example.

---

!!! failure 
    Consider a training dataset consisting of 1000 patients where we want to train a classifier to "accurately" classify whether a patient has cancer (positive class 1) or no cancer (negative class 0). The dataset is dichotimized by 950 benign patients and the remaining 50, cancerous.

    The ZeroR classifier (baseline classification model) predicts only the majority class. And in our case, the in-sample training set accuracy will be $0.95\%$ since it predicts all sample to be benign; for completeness, we also assume a validation set with 1000 patients (990 benign and 10 cancerous), it follows that our validation set's accuracy will be $0.99\%$. 

    Euphoria is at all time high over this result, surprised that a baseline model can perform so well, you then happily reported this result to your boss, and gets fired immediately. You drown in tears and googled "Is accuracy a bad metric?", only to learn a valuable lesson: since the model you trained will predict benign no matter the input, this means it completely missed out every single cancerous patient even though you get a 99 percent accuracy. Should your model ever be deployed as a "cancer detector", you will be responsible for indirectly causing many patients' death.
  
