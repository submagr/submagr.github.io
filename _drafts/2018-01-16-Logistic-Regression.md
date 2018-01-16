---
layout: post
title: Understanding Logistic Regression 
excerpt: "Yet another attempt to understand logistic regression"
modified: 2013-05-31
tags: [machine-learning, theory, experiments]
comments: true
pinned: true
image:
  feature: Log-Reg-Data.png 
---
Logistic Regression is actually a classification algorithm. It can be used to solve linearly separable problems. In logistic regression, we learn a model, which given input data, outputs label 0 or 1.

We'll understand the logistic regression with a simple case: Single Dimension linearly separable data. Suppose we have the n datapoints and we have to learn a predictor which will predict the label $$y \in \{0, 1\}$$ given $$x \in \mathbb{R}$$

Logistic regression model 
$$
\begin{gather*}
f(x|m,c) = 
\begin{cases}
  0 & sigmoid(mx+c) \ge 0.5 \\
  1 & sigmoid(mx+c) < 0.5
\end{cases}
\end{gather*}
$$
In training phase, we'll try to learn optimal m and c

## Training
For learning m and c, we have to define a loss function whose minima will give us optimal m and c. 
Cross Entropy Loss:
$$
\DeclareMathOperator{\argmin}{argmin}
\begin{equation}
\text{Loss} = \argmin_{m,c} (-1) \times \sum_{i = 1}^{N} \big\{ y_i log(\text{sigmoid}(mx_i+c)) + (1-y_i)log(1-\text{sigmoid}(mx+c))\big\}
\end{equation}
$$

For understanding the intuition behind above loss function: Suppose the label y = 1, then if prediction is 0, i.e. then loss will become $$\infty$$. This means, when we'll minimize the loss, it will try not to predict 0, when actual label is 1. Similarily for the label 0.

Note that, The cross entropy loss function is actually convex and we can use gradient descent algorithm on this to find a minima.
![](\assets\2017-12-25-G2.png)

## Experiment
Let's generate data:  
{% highlight python %}
def gen(n):
    # randomly choose a point on real line
    data = [] 
    mid = randint(-1000, 1000)
    for i in range(n):
        offset = randint(-500, 500)
        if(offset >= 0):
            data.append((mid+offset, 1))
        else:
            data.append((mid+offset, 0))
    return data, mid
{% endhighlight %}
The generated data looks like this: 

![](\assets\data-log-reg.png)

The loss function on this data looks like this:

![](\assets\cross-entropy-loss.png)

