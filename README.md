# Dimensionality Reduction

![test](https://user-images.githubusercontent.com/93261025/195637533-eeb4ad43-feb1-483e-b8ad-f64129fa3f5a.gif)

"A global geometric framework for nonlinear dimensionality reduction" 논문에서 제안된 "Swiss Roll" 데이터를 가지고 LLE, Isomap 그리고 MDS로 dimensionality reduction을 진행했을 때 양상을 살펴보겠습니다.

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
        
def plot_diagram(points, color, method, n_neighbors = 10, n_components = 2, title = '_Embedding'):
    
    if method == 'isomap':
        isomap = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components, p=1)
        transformed = isomap.fit_transform(points)
        
    elif method == 'MDS':
        md_scaling = manifold.MDS(n_components=n_components, max_iter=50, n_init=4)
        transformed = md_scaling.fit_transform(points)  
    else:
        #method : standard, ltsa, hessian, modified
        lle = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors,n_components=n_components, method=method, eigen_solver='dense')
        transformed = lle.fit_transform(points)
    plot_2d(transformed, color, f"{method}"+title)
    
def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.savefig('img/'+ title + '.png')

def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=5, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
``` 
기본 설정 (n_neighbors = 10, n_components = 2)로 dimensionality reduction 진행했을 때 plot을 확인할 수 있습니다.


### compare_neighbors()
```
def compare_neighbors(method):
    for i in range(1,10):
        plot_diagram(dataset[0], dataset[1], method, n_neighbors = i, title=f'n_neighbors_{i}')
        
def plot_graph(points, methods, n_neighbors = 10):
    num_col = 1
    num_row = 5
    fig, axs = plt.subplots(
    nrows=num_row, ncols=num_col, figsize=(7, 14), facecolor="white", constrained_layout=True
)
    fig.suptitle("Error per n_components", size=15)

    for ax, method in zip(axs.flat, methods):
        error = []
        if method == 'isomap':
            for i in range(1,4):
                isomap = manifold.Isomap(n_neighbors=n_neighbors, n_components=i, p=1)
                transformed = isomap.fit_transform(points)
                error.append(isomap.reconstruction_error())
                ax.set_ylim(0, 160)
        else:
            for i in range(1,4):
                #method : standard, ltsa, hessian, modified
                lle = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors,n_components=i, method=method, eigen_solver='dense')
                transformed = lle.fit_transform(points)
                error.append(lle.reconstruction_error_)
                ax.set_ylim(0, 0.0025)
                
        ax.plot(error)
        ax.set_title(f'{method}'.upper())
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
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
![error_n_components](https://user-images.githubusercontent.com/93261025/195656387-f4826fcf-69e7-4092-b95c-8ccdd49b0d07.png)

논문에서 제시되었던 것처럼 ISOMAP 뿐 아니라 다른 lle 기법 또한 dimension 2부터 충분히 data의 속성을 충분히 표현할 수 있음을 알 수 있습니다. 
다만 lle 기법 중 standard가 dimension이 1일 때부터 error가 거의 0으로 나타나는 양상을 보이는데, 실제로 dimension = 1로도 swiss roll 데이터가 충분히 표현될 수 있는 것인지 추가적인 확인이 필요해보입니다.

참고:
https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py

Tenenbaum, J. B., Silva, V. D., & Langford, J. C. (2000). A global geometric framework for nonlinear dimensionality reduction. Science, 290 (5500), 2319-2323.
