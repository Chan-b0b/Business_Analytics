import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from functions import *

def draw_plot(dataset):
    methods = ['standard', 'ltsa', 'hessian', 'modified','isomap','MDS']
    for method in methods:
        plot_diagram(dataset[0], dataset[1], method)

def compare_neighbors(method):
    for i in range(1,10):
        plot_diagram(dataset[0], dataset[1], method, n_neighbors = i, title=f'n_neighbors_{i}')
        
def compare_dimension(method):
    for i in range(1,4):
        plot_graph(dataset[0], method, n_components = i, title=f'n_dimension_{i}')
        
if __name__ == "__main__":
    data_name = 'swiss_roll'
    dataset = load_dataset(data_name)
    plot_3d(dataset[0],dataset[1])
    # draw_plot(dataset)
    # compare_neighbors('isomap')
    # compare_dimension('isomap')
    
    
    #https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py
    
    
    