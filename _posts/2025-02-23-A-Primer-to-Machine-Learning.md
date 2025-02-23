## A Primer to Supervised Learning in Machine Learning

<p style="font-family: Georgia, Arial, sans-serif;">
While working on machine learning problems, we often encounter the bad practice of fitting every model we know and then selecting the best-performing one—or even an ensemble—without deeper insight. This approach might work on toy examples, but it rarely succeeds on real-life datasets. In this post we will discuss the recurring theme behind the supervised learning problems and explore why a more thoughtful approach is crucial for building robust models.
</p>


---

## Understanding Supervised Learning

Let's begin by revisiting the definition of supervised learning. Machine learning can broadly be categorized into three paradigms:

    1. Supervised Learning
    2. Unsupervised Learning
    3. Reinforcement Learning

It is not always clear which of these classes a problem falls into, it often depends on how we formulate the problem rather than how it initially appears. In this post, we will focus solely on the supervised learning setup. 

## What Makes Learning "Supervised"?

Loosely speaking, a supervised problem involves analyzing a phenomenon (i.e., the target variable) based on some pre-identified predictors (i.e., features). However, there is a catch, [correlation doesn't imply causation.](https://simple.wikipedia.org/wiki/Correlation#Correlation_vs_causation)

Some common instances of supervised learning problems are:

- Forecasting rainfall based on temperature, humidity, wind speed, etc.
- Identifying spam emails based on the presence of predefined keywords or patterns.
- (For Advanced Readers) Training decoder-based Large Language Models is also an example of a supervised learning framework; these models are trained to predict the next token in a sentence—and this setup works amazingly well.

## Types of Supervised Learning

Supervised learning is further classified based on the nature of the target variable:

- Regression: When the target variable is continuous.
- Classification: When the target variable is discrete.

`The goal of categorizing problems into multiple types is that no single algorithm works universally well. If that were the case, life would be so simple—but unfortunately, it's not. Each algorithm caters to specific types of use cases.`

## Mathematical Framework

With this definition of supervised learning, let's look at underlying mathematical framework to tackle such problems:

Let, $$Y$$ denote the response random variable(roughly, a random variable can be thought of a function( a measurable function, don't worry if you don't understand this term it won't matter in our discussions) from sample space to $$\mathbb{R^d}$$) and $$X$$ be the feature random variable(r.v.).

Our goal is to understand the relation between $$X$$ and $$Y$$.

Let, $$f:X\to Y$$ be the function that defines the relation between $$X$$ and $$Y$$. 

...but, how to choose $$f~?$$ 
We must choose $$f$$ such that it minimizes the expected prediction error.
For regression this turned out to be minimizing, $$\text{EPE}(f) = E(f(X)-Y)^2$$ which can be written as,

$$ EPE(f) = E_{X}E_{Y|X}(f(X)-Y)^2 $$

which is same as finding point wise minimum i.e. finding best value for $$f(\tilde{c})$$ for each $$X = \tilde{c}.$$ So, our task is to minimize $$E\[(f(\tilde{c})-Y)^2|X=\tilde{c}\]$$ which is given by , $$f(\tilde{c}) = E(Y|X=\tilde{c}).$$

Therefore, best $$f$$ is the conditional mean of $$Y$$ given $$X.$$

However, in real scenarios, we rarely have enough data to compute the exact mean of 
$$Y$$ for each $$X.$$ Therefore, we often make assumptions about the structure of 
$$f$$ globally (as in linear regression or logistic regression) or locally (as in k-nearest neighbors or decision trees). 

We will explore these approaches in detail in future posts.
