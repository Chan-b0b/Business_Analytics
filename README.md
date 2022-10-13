# Dimensionality Reduction

![test](https://user-images.githubusercontent.com/93261025/195637533-eeb4ad43-feb1-483e-b8ad-f64129fa3f5a.gif)

Tenenbaum, J. B., Silva, V. D., & Langford, J. C. (2000). A global geometric framework for nonlinear dimensionality reduction. Science, 290 (5500), 2319-2323. 논문에서 제안된 "Swiss Roll" 데이터를 가지고 LLE, Isomap 그리고 MDS로 dimensionality reduction을 진행했을 때 양상을 살펴보겠습니다.

### Dataset
Dataset은 sklearn에서 제공하는 swiss_roll을 사용합니다.
data_name을 'make_s_curve' 등 다양하게 변경할 수 있습니다
``` 
data_name = 'swiss_roll'
dataset = load_dataset(data_name)
``` 
### Methods
사용 가능한 method는 총 6가지가 있습니다.
methods = ['standard', 'ltsa', 'hessian', 'modified','isomap','MDS']

## main.py

main.py에서 아래 함수 중 선택할 수 있습니다


### draw_plot()

![test](https://user-images.githubusercontent.com/93261025/195640342-8c68ca55-9ce3-44e0-963e-f4e7066f1b7b.png)

``` 
def draw_plot(dataset):
    methods = ['standard', 'ltsa', 'hessian', 'modified','isomap','MDS']
    for method in methods:
        plot_diagram(dataset[0], dataset[1], method)
``` 
기본 설정 (n_neighbors = 10, n_components = 2)로 dimensionality reduction 진행했을 때 plot을 확인할 수 있습니다.


### compare_neighbors()
```
def compare_neighbors(method):
    for i in range(1,10):
        plot_diagram(dataset[0], dataset[1], method, n_neighbors = i, title=f'n_neighbors_{i}')
```
![standard_neighbors](https://user-images.githubusercontent.com/93261025/195644353-25a26058-8e81-448d-bfe3-1af4c7fcca7d.gif)
![neighbors](https://user-images.githubusercontent.com/93261025/195642274-63a72f00-3c89-476b-a7a9-b96e6dd2ef6a.gif)

Neighbor의 수가 달라짐에 따라 plot되는 양상의 변화를 확인할 수 있습니다.
        
### compare_components()        
```
def compare_components(method):
    for i in range(1,4):
        plot_graph(dataset[0], method, n_components = i, title=f'n_dimension_{i}')
```
