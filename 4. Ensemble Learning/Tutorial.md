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

### Square root of total parameters
It is known that the performance of random forests gets better if we choose only a portion of features from the dataset. Square root of the number of total parameters is usually used. Although professor Kang gave us an excellent example explaining why this could be possible, but I wanted to test by myself to see whether if it's true or not.

**Result**

The result of difference in number of features is as follows. We only showed the result of Income and Titanic datasets because the result for Bankrupcy and Wine datasets were not showing difference in this case. For the rest datasets, they were not showing huge difference gap neither, but in case of the Income dataset, the performace was slightly better if we used only the square root number of attributes. This complies with what we have learned in the class. However, in case of Titanic dataset, using all the parameters given was showing a slightly better performance. Even if we consider such difference as negligible, we were able to find that limiting the number of parameters does not incur hugh difference in performance. 

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
  
  ```
  for datum in data:
    print(f'-------------{datum["name"]}---------------')
    df = datum['data']
    y_name = datum['y']
    df = df.dropna(axis='index')
    X = df.drop([y_name], axis = 1)
    y = df[y_name]
    
    features_to_encode = X.columns[X.dtypes==object].tolist() 
    
    col_trans = make_column_transformer(
                        (OneHotEncoder(),features_to_encode),
                        remainder = "passthrough"
                        )    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)

    model = RandomForestClassifier(random_state=1,max_features=None)
    pipe = make_pipeline(col_trans, model)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred))
    
        
    model2 = RandomForestClassifier(random_state=1)
    pipe = make_pipeline(col_trans, model2)

    pipe.fit(X_train, y_train)
    y_pred2 = pipe.predict(X_test)
    print(classification_report(y_test, y_pred2))
    ```

### Deciding number of features
We have checked that limiting the number of features to the square root of number of total parameters does not always lead to best result, so we are trying to check the performance gap while changing the fraction of parameters we select. 

**Result**

- Income

![income](https://user-images.githubusercontent.com/93261025/205084016-0bc3d12e-0bdb-4a42-bf30-25afff3253ca.png)

- Bankrupcy

![Bankruptcy](https://user-images.githubusercontent.com/93261025/205084052-b9dec477-ce4d-45e8-9780-27939a63ee04.png)

- Wine

![Wine](https://user-images.githubusercontent.com/93261025/205084091-c23cdf92-3764-4a3b-b3f2-339af12f0e4d.png)

- Titanic

![titanic](https://user-images.githubusercontent.com/93261025/205084116-82b06169-8d1e-4b2d-b116-1f131010062f.png)

```Python
for datum in data:
    output = []
    for fraction in [0.1*x for x in range(1,10)]:
        df = datum['data']
        y_name = datum['y']
        df = df.dropna(axis='index')
        X = df.drop([y_name], axis = 1)
        y = df[y_name]

        features_to_encode = X.columns[X.dtypes==object].tolist() 

        col_trans = make_column_transformer(
                            (OneHotEncoder(),features_to_encode),
                            remainder = "passthrough"
                            )    

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)

        model = RandomForestClassifier(random_state=1,max_features=fraction)
        pipe = make_pipeline(col_trans, model)

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        output.append(classification_report(y_test, y_pred,output_dict=True)['weighted avg']['f1-score'])

    plt.plot(output)
    plt.plot(output, color='#e35f62', marker='*', linewidth=2)
    plt.xticks([x for x in range(9)], labels = [round(0.1*x,2) for x in range(1,10)])
    plt.savefig(f'{datum["name"]}.png')
    plt.clf()
```

### Deciding criterion
Random Forest may construct different decision tree depending on which criterion it uses. The default setting is gini impurity, and also entropy is also available.

**Result**
The result has shown that there isn't any significant difference in performance occured by changing the criterion. In case of Income dataset, using the gini score shows higher performance, whereas entropy showed higher performance with Titanic dataset. 

  | **Criterion** | Income | Bankruptcy | Wine | Titanic |
  | --- | --- | --- | --- | --- |
  | Gini | 0.848 | 1.0 | 0.978 | 0.803 |
  | Entropy | 0.847 | 1.0 | 0.978 | 0.804 |

### Result
- Experiment 1 : **Square root of total parameters**
   - Limiting to the square root of the number of total parameters did not always lead to higher performance, and even when it makes a difference, the performance gap was negligible.
 
- Experiment 2 : **Deciding number of features**
   - The optimal fraction of parameters was different depending on datasets. Also, we were able to observe that the performance does not change monotonically as the fraction increases
 
- Experiment 3 : **Deciding criterion**
   - Whether choosing the criterion as gini or entropy did not occur noticeable performance difference.
