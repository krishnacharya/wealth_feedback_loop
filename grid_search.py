#importing my scripts
from objective import *
from plot_handling import *
from sigmoid_wealth_function import *
from helpers import *
from paramclass import *

#library imports
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm #progress bar graphic
from sklearn.model_selection import ParameterGrid #for grid search

def grid_dump(filename, alpha_step=0.1, gammma_step=0.1, sigma_step=0.1, threshold_step=0.1, lamb_step=0.1, df_step=0.1, granu = 10):
  '''
  About the local variables:
    sig_grid denotes the grid of parameters for the sigmoid wealth curve
    obj_grid denotes the lambda and discount factor grid

  About the Inner functions:
    get_grid_forsigmoid()
    get_grid_forobj()
    are written for neat seperation of non essential code from the Grid search that occurs in the double for loop
  '''
  def get_grid_forsigmoid():
    '''
    Returns a grid for alpha, gamma, sigma, threshold values
    Note these 4 parameters generate a unique sigmoid wealth curve
    '''
    param_grid = {'alpha': np.arange(0+alpha_step, 1+alpha_step, alpha_step), 
                'gamma': np.arange(0+gammma_step,1+gammma_step,gammma_step),
                'sigma': np.arange(0+sigma_step,1+sigma_step,sigma_step),
                'threshold': np.arange(0+threshold_step, 1+threshold_step,threshold_step)
               }
    return ParameterGrid(param_grid)
  def get_grid_forobj():
    '''
    Lambda and Discount factor grid, objective parameters

    Recall that objective = sum df^(t-1) (lambda c + (1-lambda) (distance to x_2))
    '''
    param_grid = {'lamb': np.arange(0, 1+lamb_step, lamb_step),
                  'df' : np.arange(0, 1+df_step, df_step),
                }
    return ParameterGrid(param_grid)

  sig_grid = get_grid_forsigmoid()
  obj_grid = get_grid_forobj()
  results_array = []
  #GRID SEARCH
  for sig_params in tqdm(sig_grid): #tqdm just records the progress bar on the terminal
    alpha, gamma, sigma, threshold = sig_params['alpha'], sig_params['gamma'], sig_params['sigma'], sig_params['threshold']
    if threefixedpoints(alpha, gamma, sigma, threshold): #if we dont have 3 fixed point just skip
      x_1, x_2, x_3 = get_fixedpoints(alpha, gamma, sigma, threshold)
      Delta = get_Delta(alpha, gamma, sigma, threshold) #Fixed points and Delta are independent of discount factor and lambda
      cost_vec = get_cost_vec(x_1, x_2, Delta, granu = granu)
      for obj_params in obj_grid:
        lamb, df = obj_params['lamb'], obj_params['df']
        value_object = Paramclass(alpha, gamma, sigma, threshold, lamb, df) #creating an object of paramclass type
        value_object.set_FP_Delta(x_1, x_2, x_3, Delta)
        value_object.set_cost_vec(cost_vec)
        value_object.set_obj()
        results_array.append(value_object)
  np.save(filename, results_array)