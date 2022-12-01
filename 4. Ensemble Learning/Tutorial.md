# Ensemble Learning

This tutorial is for Chapter 4_Ensemble Learning of "Business Analytics" class in Industrial Engineering Department in Korea University

## Random Forest
Random Forest is one of the models that you must have heard at least once if you are in the field of machine learning. I myself have seen this model performing in a fair performance even though the model itself remains comparatively in a simple form. For this reason, I'll be performing tests 

## Background
**Reference**

[https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/](https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/)

![image](https://user-images.githubusercontent.com/93261025/204733447-d073df62-ebc9-4a01-99dc-0a0b623c5070.png)

**Steps involved in random forest algorithm:**

**Step 1:** In Random forest n number of random records are taken from the data set having k number of records.

**Step 2:** Individual decision trees are constructed for each sample.

**Step 3:** Each decision tree will generate an output.

**Step 4:** Final output is considered based on Majority Voting or Averaging for Classification and regression respectively.

**Important Features of Random Forest**
- 1. Diversity- Not all attributes/variables/features are considered while making an individual tree, each tree is different.
- 2. Immune to the curse of dimensionality- Since each tree does not consider all the features, the feature space is reduced.
- 3. Parallelization-Each tree is created independently out of different data and attributes. This means that we can make full use of the CPU to build random forests.
- 4.  Train-Test split- In a random forest we don’t have to segregate the data for train and test as there will always be 30% of the data which is not seen by the decision tree.
- 5.  Stability- Stability arises because the result is based on majority voting/ averaging.

**Difference Between Decision Tree & Random Forest**
| Decision Tree | Random Forest |
| --- | --- |
| 1. Decision trees normally suffer from the problem of overfitting if it’s allowed to grow without any control. | 1. Random forests are created from subsets of data and the final output is based on average or majority ranking and hence the problem of overfitting is taken care of. |
| 2. A single decision tree is faster in computation. | 2. It is comparatively slower. | 
| 3. When a data set with features is taken as input by a decision tree it will formulate some set of rules to do prediction. |3. Random forest randomly selects observations, builds a decision tree and the average result is taken. It doesn’t use any set of formulas. |

**Important Hyperparameters**

Hyperparameters are used in random forests to either enhance the performance and predictive power of models or to make the model faster.

Following hyperparameters increases the predictive power:

1. **n_estimators**– number of trees the algorithm builds before averaging the predictions.

2. **max_features**– maximum number of features random forest considers splitting a node.

3. **mini_sample_leaf**– determines the minimum number of leaves required to split an internal node.

Following hyperparameters increases the speed:

1. **n_jobs**– it tells the engine how many processors it is allowed to use. If the value is 1, it can use only one processor but if the value is -1 there is no limit.

2. **random_state**– controls randomness of the sample. The model will always produce the same results if it has a definite value of random state and if it has been given the same hyperparameters and the same training data.

3. **oob_score** – OOB means out of the bag. It is a random forest cross-validation method. In this one-third of the sample is not used to train the data instead used to evaluate its performance. These samples are called out of bag samples.

--------
## Dataset

We have chosen four datasets in total, which are adult_income, bankruptcy, wine, and titanic dataset. These datasets are well-known datasets used for testing performance of classification models.

```
col_name = ['age','workclass','fnlwgt','education','education_num','marital_status','occupation',
            'relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country',
            'high_income']
income = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
            names=col_name)
income = {"data":income, "y":'high_income',"name" : 'income'}

col_name = ['Industrial Risk','Management Risk','Financial Flexibility', 'Credibility', 'Competitiveness', 'Operating Risk', 'Class']
Bankruptcy =  pd.read_csv('data/Qualitative_Bankruptcy/Qualitative_Bankruptcy.data.txt', sep=",",names=col_name)
Bankruptcy = {"data":Bankruptcy, "y":'Class',"name" : 'Bankruptcy'}

col_name = ['class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
Wine =  pd.read_csv('data/wine/wine.data', sep=",", names = col_name)
Wine = {"data":Wine, "y":'class',"name" : 'Wine'}

titanic =  pd.read_csv('data/titanic/train.csv')
titanic = titanic.loc[:,['Survived','Pclass','Sex','Age','Parch','Fare','Embarked']]
titanic = {"data":titanic, "y":'Survived',"name" : 'titanic'}


data = [income, Bankruptcy, Wine, titanic]

```

## Experiment

### Deciding number of features
It is known that the performance of random forests gets better if we choose only a portion of features from the dataset. Square root of the number of total parameters is usually used. Although professor Kang gave us an excellent example explaining why this could be possible, but I wanted to test by myself to see whether if it's true or not.

**Result**
The result of difference in number of features is as follows. We only showed the result of Income and Titanic datasets because the result for Bankrupcy and Wine datasets were not showing difference by such 

- Income

  | **Income_all** | Precision | Recall | F1-Score | 
  | --- | --- | --- | --- |
  | <=50K | 0.88 | 0.92 | 0.90 |
  | >50K | 0.72 | 0.62 | 0.67 |
  | Accuracy | --- | --- | 0.85 |
  | Macro Avg | 0.80 | 0.77 | 0.78 |
  | Weighted Avg | 0.84 | 0.85 | 0.84 |

  | **Income_sqrt** | Precision | Recall | F1-Score | 
  | --- | --- | --- | --- |
  | <=50K | 0.88 | 0.93 | 0.90 |
  | >50K | 0.73 | 0.62 | 0.67 |
  | Accuracy | --- | --- | 0.85 |
  | Macro Avg | 0.81 | 0.78 | 0.79 |
  | Weighted Avg | 0.84 | 0.85 | 0.85 |
  
- Titanic

  | **Titanic_all** | Precision | Recall | F1-Score | 
  | --- | --- | --- | --- |
  | 0 | 0.85 | 0.81 | 0.83 |
  | 1 | 0.75 | 0.80 | 0.77 |
  | Accuracy | --- | --- | 0.80 |
  | Macro Avg | 0.80 | 0.80 | 0.80 |
  | Weighted Avg | 0.81 | 0.80 | 0.80 |
  
  | **Titanic_sqrt** | Precision | Recall | F1-Score | 
  | --- | --- | --- | --- |
  | 0 | 0.83 | 0.84 | 0.83 |
  | 1 | 0.77 | 0.76 | 0.76 |
  | Accuracy | --- | --- | 0.80 |
  | Macro Avg | 0.80 | 0.80 | 0.80 |
  | Weighted Avg | 0.80 | 0.80 | 0.80 |
