---
title: Impact of AI on Students
feed: show
date: 2026-07-21
---

The goal of this project is to explore the effects of AI on students. The data is available on [Kaggle](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students) and it's entirely synthetic. The aim is to

- Perform EDA
- Train a model to predict `Post_Semester_GPA` (regression)
- Train a model to predict `Burnout_Risk_Level`, with at least $90\%$ recall on the `High` class so that those students can receive proper care

# EDA

The data is composed of the following columns

| Column Name | Possibilities |
|----|----|
| Student_ID | Number $100001–150000$ (int) |
| Major_Category | STEM, Business, Humanities, Medical, or Arts |
| Year_of_Study | Freshman, Sophomore, Junior, Senior, or Graduate |
| Pre_Semester_GPA | Number in $1.0–4.0$ (float) |
| Weekly_GenAI_Hours | Number in $0.0–40.0$ (float) |
| Primary_Use_Case | Copywriting, Summarizing, Debugging, Ideation, or Direct Answer |
| Prompt_Engineering_Skill | Beginner, Intermediate, or Advanced |
| Tool_Diversity | Scale $1$ to $5$ (int) |
| Paid_Subscription | True or False |
| Traditional_Study_Hours | Number in $1.0–36.0$ (float) |
| Perceived_AI_Dependency | Scale $1$ to $10$ (int) |
| Institutional_Policy | Allowed With Citation, Strict Ban, or Actively Encouraged |
| Anxiety_Level_During_Exams | Scale $1$ to $10$ (int) |
| Post_Semester_GPA | Number in $1.0–4.0$ (float) |
| Skill_Retention_Score | Number in $0.0–100.0$ (float) |
| Burnout_Risk_Level | Low, Medium, High |

Some of the columns have **categorical data** which has to be converted to numbers to be analyzed.
- The `Student_ID` was completely ignored for the rest of our analyses as it is irrelevant.
- The columns `Year_of_Study`, `Burnout_Risk_Level`, `Prompt_Engineering_Skill` have some sort of ordering. E.g. for `Burnout_Risk_Level`, we have that `Low` \< `Medium` \< `High`, so we can assign `1`, `2`, `3`, respectively. The same was done for the other columns, starting at `0` and going up to `#possibilities - 1`
- The columns `Major_Category`, `Primary_Use_Case` and `Institutional_Policy` have categorical data with no obvious ordering, so we used dummy/indicator values (one-hot encoding). It consists of creating one additional column for each possibility in that class, and then assigning `1` for that specific class possibility in each row entry, while keeping the others at `0`.
- We created two data columns, `Diff_GPA` = `Post_Semester_GPA - Pre_Semester_GPA` and `Diff_GPA_Relative` = `(Post_Semester_GPA - Pre_Semester_GPA) / Pre_Semester_GPA` that help us visualise the impact of AI.

Here I show a scatter plot of all the data, the diagonal plots have the distributions of the data

![[274359e43eee2f97411114ef97b006862d06af79.png|600]]

The correlation matrix of all the data is

![[b0724d6b0758bd4ebc94386d45adf83015df9d92.png|600]]

## `Post_Semester_GPA`

In both plots, it is pretty obvious that there is a large correlation between `Pre_Semester_GPA` and `Post_Semester_GPA`, which is great news for our goal to perform regression. It also looks like there is a tiny correlation between `Diff_GPA` and `Traditional_Study_Hours` which will also help regression.

## `Burnout_Risk_Level`

It is very difficult to see anything interesting for the `Burnout_Risk_Level` except that, looking at its histogram, the classes seem somewhat balanced.

## Clean up the data

After some trial and error, it became obvious that the data has some weird excesses for which we have no explanation and which, we suspect, are not trustworthy and should be dropped.

### Excess in `Weekly_GenAI_Hours`

The first excess appeared in `Weekly_GenAI_Hours`. In a semi-log histogram, we have

![[346f004a447305d69a03b1ea43fe80afd7d68b2b.png|500]]

where we can see $4$ weird excesses at `Weekly_GenAI_Hours` \> 30. The bin count also seems linear, so we suspect that this distribution is an exponential distribution. After doing an exponential fit, we determined that its distribution function is approximately

$$p(x) = \frac{1}{8.4277522}e^{-x/8.4277522},$$
To calculate the locations of the excesses, we calculated the expected number of counts per bin (of the histogram) and performed a binomial test to calculate the $p$-value. The bin $p$-value is ![[0a9250403ac1d20eeee568b2f51783319fd738a1.png|500]]
where we can clearly see four large excesses. The red line is $p$-value \< 0.05 (accounting for the look elsewhere effect of having $300$ bins).

We found that the excesses were exactly at `Weekly_GenAI_Hours` = $34$, $36$, $38$ and $40$, so these points were removed.

Its exponential distribution also makes some models fit worse, so we applied a $x \to \log(1 + x)$ transform to it. The final distribution is

![[8a4ba93d81e62933bcfecc971f9c191e3c3eb7f2.png|500]]

### Excess in `Traditional_Study_Hours`

There is also a very large excess at `Traditional_Study_Hours` exactly $1$.

![[a168523e781c1d01afb70247b2f450c6083fef33.png|500]]

These points look like a relic from the data being synthetic and a lower bound being imposed on `Traditional_Study_Hours`. So we removed them.

### Excess in `Skill_Retention_Score`

There is also a very large excess at `Skill_Retention_Score` exactly $100$.

![[90cc12e6d3f9bb528d9a9b2b1cb8358b77c80c78.png|500]]

These points look like a relic from the data being synthetic and an upper bound being imposed on `Skill_Retention_Score`. So we removed them.

## EDA on `Post_Semester_GPA`

Making a scatter plot of `Pre_Semester_GPA` vs `Post_Semester_GPA` shows a very clear linear relation. The diagonal line is $y=x$ and, as most of the points are above the line, this shows that `Post_Semester_GPA > Pre_Semester_GPA`, so the grades improved.

![[91c71c32d7f24cddd8f904f47d476f12095d2f4d.png|500]]

Showing a plot of `Pre_Semester_GPA` vs `Diff_GPA_Relative`, with a red line at $y=0$, we can see that `Diff_GPA_Relative` is mostly positive, so grades improved. It also has a very tiny downward trend, which indicates that the higher your `Pre_Semester_GPA`, the less you improved. This is a consequence of the upper bound at $4$, and the fact that it gets harder and harder to improve. Still, this effect is very small.

![[84ee2c837283e39faa9f033b58013f264c13914b.png|500]]

Showing the scatter plot of `Traditional_Study_Hours` vs `Diff_GPA_Relative`, we see an upward trend: a correlation where the more you study the more your grades increase, which is intuitive.

![[e496f83b3dc3b2d614c9aca9a3fbf26976df066b.png|500]]

## EDA on `Burnout_Risk_Level`

The class count is

| Categorical | Value | Count  | Fraction of total |
|-------------|-------|--------|-------------------|
| Low         | 0     | 15 391 | 0.334             |
| Medium      | 1     | 19 682 | 0.427             |
| High        | 2     | 10 985 | 0.239             |

so the classes are more or less balanced.

In the following plot, I show a histogram of `Diff_GPA_Relative` grouped by `Burnout_Risk_Level`. We can see no big difference between the `Burnout_Risk_Level`. I checked other features and none produced any clear trend.

![[6357775e01124c2f87bd8dcb23f9c11b2d4953d2.png|500]]

# Predicting `Post_Semester_GPA` (regression)

The goal of this section is to predict `Post_Semester_GPA` using the `RMSE` metric.

## Feature selection

We already suspect `Pre_Semester_GPA` and `Traditional_Study_Hours` to be excellent predictors but, as we do not want to drop any important feature, we are going to employ some feature selection methods.

1.  We did `RFECV` using a boosted decision tree with `max_depth = 6`
2.  We did `RFECV` using a random forest with `max_depth = 6`
3.  We did `Tree-Based Feature Importance` using random forests with `max_depth = 6` and kept all features until we crossed $0.99$ importance.

These models were selected through trial and error in trying to train the data using whatever features looked important. We excluded `Post_Semester_GPA`, `Burnout_Risk_Level`, `Diff_GPA` and `Diff_GPA_Relative` from the available choices.

We combined all three methods and selected a feature if at least two methods suggested it. We arrived at

- `Pre_Semester_GPA`
- `Traditional_Study_Hours`
- `Year_of_Study`
- `Weekly_GenAI_Hours`
- `Prompt_Engineering_Skill`
- `Debugging/Troubleshooting`
- `Direct_Answer_Generation`

## Models

We trained a large set of models using K-fold (k=5) to reduce sampling bias. The models selected were

- **Linear Regression**
- **KNN** - neighbors = 3, 5, 7, 9, 11, 15, 21, 23, 30, 35, 40; weights = uniform, distance; metric = euclidean, manhattan, minkowski
- **Random Forest** - n_estimators = 50, 100; max_depth = 5, 10, 20; min_samples_split = 2, 10; min_samples_leaf = 1, 4; max_features = sqrt, log2
- **Boosted Decision Trees** - n_estimators = 50, 100; max_depth = 3, 5, 10, 20; learning_rate = 0.05, 0.1
- **SVR** - C = 0.1, 1, 10; kernel = rbf, linear; gamma = scale, auto, 0.1, 1

## Results

Based on the `RMSE` error, the top-20 models are

![[e255747b91b2da997d1382575eb284b575e1e262.png]]

with the best model being, by a tiny margin, an SVR with `C = 1`, `gamma = 0.1` and an `rbf` kernel.

The best model has a median `RMSE` of around `0.145` on predicting `Post_Semester_GPA`, which has a `3.34` average. This means that the `RMSE` is `4.3%` of the average. Looking at the `MAPE` metric (average of relative errors), this model's median `MAPE` lands at `3.555%`, which is consistent.

A linear fit also seemed to perform very well, with a `RMSE` of `0.1585` and a `MAPE` of `3.875%`.

# Predicting `Burnout_Risk_Level` (classification)

The goal of this section is to predict `Burnout_Risk_Level` according to the metric `accuracy`. We also do threshold tuning to make sure we catch most of the `High` cases, i.e. we require the `recall` of that class to be at least `0.9`.

## Feature selection

We performed three feature selection algorithms

1.  We did `RFECV` using a boosted decision tree with `max_depth = 6`
2.  We did `RFECV` using a random forest with `max_depth = 6`
3.  We did `Tree-Based Feature Importance` using random forests with `max_depth = 6` and kept all features until we crossed $0.99$ importance.

We combined all three methods and selected a feature if at least two methods suggested it. We arrived at

- `Year_of_Study`
- `Pre_Semester_GPA`
- `Weekly_GenAI_Hours`
- `Traditional_Study_Hours`
- `Perceived_AI_Dependency`
- `Strict_Ban`
- `Anxiety_Level_During_Exams`

## Models

We trained a large set of models using K-fold (k = 5) to reduce sampling bias. The models selected were

- **KNN** --- neighbors = 3, 5, 7, 9, 11, 15, 21, 23, 30, 35, 40; weights = uniform, distance; metric = euclidean, manhattan, minkowski
- **Random Forest** --- n_estimators = 50, 100; max_depth = 5, 10, 20; min_samples_split = 2, 10; min_samples_leaf = 1, 4; max_features = sqrt, log2
- **Boosted Decision Trees** --- n_estimators = 50, 100; max_depth = 3, 5, 10, 20; learning_rate = 0.05, 0.1
- **SVC** --- C = 0.1, 1, 10; kernel = rbf, linear; gamma = scale, auto, 0.1, 1
- **NN** - AF = Relu, ELU; hidden layers = 2, 4, 6; drop out = 0, 0.2, 0.4; neurons hidden layers = 4, 16, 64. The optimizer used was AdamW with lr = 1e-3.

## Results

In the next plot, I show the top-20 performing models, according to their `accuracy` which is defined as $$\text{accuracy} \equiv \frac{\text{True Positives} + \text{True Negatives}}{\text{True Positives} + \text{False Negatives} + \text{False Positives} + \text{True Negatives}} = \frac{\text{Correct}}{\text{All}}$$

In the following plot, I show the class `f1_score`, defined as the harmonic average of precision and recall, defined as
$$\text{F1}\equiv \frac{2}{1/\text{precision} + 1/\text{recall}},$$ where precision and recall are defined as
$$\text{precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}},$$
$$\text{recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}.$$

Basically, `precision` tells you whether you can trust your model when it predicts a specific class. And recall tells you how many class elements were successfully classified.

![[91ee992b6739198819cebbc8795c9c3bb876d6db.png]]

Although some models have very slightly better scores than others, the differences fall within the sampling variation and therefore there is no clear winner. For the following section, we will take the NN model with $2$ hidden layers, each with $512$ neurons, no dropout and with a `ReLU` activation function as our benchmark model.

In binary classification, an accuracy of around $0.5$ would be disastrous as it would be no better than random chance. Since we have $3$ target classes, random chance would yield $1/3$ accuracy, so an accuracy of around $0.52 - 0.54$ shows that the model learned some of the features. That being said, these models are still considered rather poor at classifying `Burnout_Risk_Level`.

In the following plot, we plot each class's `f1-score` for the top-20 accuracy performers, which is defined as
$$\text{F1}\equiv \frac{2}{1/\text{precision} + 1/\text{recall}},$$ where precision and recall are defined as
$$\text{precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}},$$
$$\text{recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}.$$

`precision` $\to$ how many predicted elements of that class actually belong to that class
`recall` $\to$ how many elements of that specific class were successfully classified.

From the plot, we can see that `High` seems to have a slightly lower `f1-score`, so it's slightly less performant, while for the other two it's a mixed bag.

![[7293eee24a4805703474d6799a7eebf02c4caf3a.png]]

## Threshold tuning

For many scenarios, the cost of mispredicting a class can be very high and, for that reason, it's worth trading some model accuracy for recall in a particular class. For our scenario, it is important to detect students close to burnout, so that they can get proper treatment/help.

We will tune the threshold so that we catch at least $90\%$ of the students with a `High` risk of burnout.

In contrast to binary classification, we have three classes and there is no clear threshold that some class must cross in order to be classified as that particular class. Our approach was to create a `threshold` specifically for the `High` class which, if crossed, selects the `High` class. If the threshold is not crossed, we select the class with the highest probability.

Some cases are:

- `threshold` = 0 - We always predict the `High` class.
- `threshold` = 0.1 - If the probability of being `High` is \> 0.1, we select `High`
- `threshold` \> 0.5 - We always select the class with the highest probability.

For that last reason, the plot plateaus at threshold $= 0.5$, where we get the accuracy found during training. We calculated the `High` recall and accuracy for the test set of each fold.

![[c4fed7ac4893c2c0b73f078fc5fe3d5899fa8abb.png]]

To get a `High` recall of $0.9$, we found that the threshold should be around
$$\text{threshold} = 0.115 \pm 0.003,$$
and, at this `threshold`, the model's accuracy is roughly
$$\text{accuracy} = 0.43 \pm 0.01.$$
This is around $20\%$ lower than the model without threshold, but still above $\frac{1}{3}$ (random chance).

We could also have combined the `Low` and `Medium` classes from the start, performed binary classification and binary threshold tuning. This would have simplified a lot of the methods and maybe increased performance. This was not done because it might be useful to know which students are in the `Medium` class, so that they can also be helped. It's just that misclassifying a `High` student is worse than misclassifying a `Medium` student.
