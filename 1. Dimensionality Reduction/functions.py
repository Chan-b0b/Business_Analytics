from numpy.random import RandomState
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn import manifold, datasets
# from matplotlib import gridspec
# from celluloid import Camera
# from moviepy.editor import *
from matplotlib import animation

def load_dataset(data_name):
    rng = RandomState(0)
    n_samples = 1500
    if data_name == 'swiss_roll':
        S_points, S_color = datasets.make_swiss_roll(n_samples, random_state=rng)
    elif data_name == 's_curve':
        S_points, S_color = datasets.make_s_curve(n_samples, random_state=rng)
        
    return S_points, S_color

def plot_diagram(points, color, method, n_neighbors = 10, n_components = 2, title = '_Embedding'):
    
    if method == 'isomap':
        isomap = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components, p=1)
        transformed = isomap.fit_transform(points)
        plot_2d(transformed, color, f"{method}"+title)
    elif method == 'MDS':
        md_scaling = manifold.MDS(n_components=n_components, max_iter=50, n_init=4)
        transformed = md_scaling.fit_transform(points)  
        plot_2d(transformed, color, f"{method}"+title)
    else:
        #method : standard, ltsa, hessian, modified
        lle = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors,n_components=n_components, method=method)
        transformed = lle.fit_transform(points)
        plot_2d(transformed, color, f"lle_{method}"+title)

        
    

def plot_3d(points, points_color):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    col = ax.scatter(x, y, z, c=points_color, s=5, alpha=0.8)
    # ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(3))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(3))
    
    # camera = Camera(fig)
    # plt.savefig('img/'+ title + '.png')

    # for ii in range(0,360,1):
    #     ax.view_init(azim=ii, elev=9)
    #     camera.snap()
    # animation = camera.animate()
    # animation.save('test.avi')
    # clip = (VideoFileClip("test.avi").speedx(2))
    # clip.write_gif("test.gif")
    def init():
        ax.scatter(x, y, z, c=points_color, s=5, alpha=0.8) 
        return fig,

    def animate(i):
        ax.view_init(elev=9., azim=i)
        return fig,

# Animate
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=360, interval=20, blit=True)
    anim.save('test.mp4', fps=30)

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

def plot_graph(points, method, n_neighbors = 10):
    errors = []
    
    if method == 'isomap':
        for i in range(1,4):
            isomap = manifold.Isomap(n_neighbors=n_neighbors, n_components=i, p=1)
            transformed = isomap.fit_transform(points)
            errors.append(isomap.reconstruction_error())
        fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
        ax.plot(errors)
        plt.savefig('img/isomap_error.png')
    
    else:
        methods = ['standard', 'ltsa', 'hessian', 'modified']
        fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
        error = []
        for method in methods:
            for i in range(1,4):
                lle = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors,n_components=i, method=method)
                transformed = lle.fit_transform(points)
                error.append(lle.reconstruction_error())
            ax.plot(errors, label = f'{method}')