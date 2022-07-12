import matplotlib.pyplot as plt
import numpy as np
from sigmoid_wealth_function import *
def save_fixedpoint_plot(alpha, gamma, sigma, threshold, new_folder):
  '''
      Just plot y=x, y=f_mu(alpha, gamma, sigma, threshold)
      Note: this plot is for when we have 3 fixed points (but it also works for other degenerate cases)    
  '''
  ss = 0.01 #stepsize for x_values in [0,1]
  x_vals = np.arange(0,1+ss,ss)
  f_vals = [f_mu(x, alpha, gamma, sigma, threshold) for x in x_vals]
  plt.plot(x_vals, x_vals, label = 'y=x', color = 'blue') # y = x in blue
  plt.plot(x_vals,f_vals, label = 'f(x)', color = 'red') # f_mu(x) in red
  plt.axvline(x = threshold/(1-alpha), color = 'yellow') # inflection point vertical line in yellow

  x_star1,x_star2 = get_extrema_points(alpha, gamma, sigma, threshold)
  x_1, x_2, x_3 = get_fixedpoints(alpha, gamma, sigma, threshold)

  plt.axvline(x = x_1, color = 'green') # 3 fixed points in green
  plt.axvline(x = x_2, color = 'green')
  plt.axvline(x = x_3, color = 'green')
  plt.axvline(x = x_star1, color = 'black') # first extrema of g(x) = x-f(x)
  plt.axvline(x = x_star2, color = 'black') # second extrema
  plt.grid()
  plt.savefig(new_folder + "/FPplot.png")
  plt.clf()

def display_fixedpoint_plot(alpha, gamma, sigma, threshold):
  '''
      Just plot y=x, y=f_mu(alpha, gamma, sigma, threshold)
      Note: this plot is for when we have 3 fixed points (but it also works for other degenerate cases)    
  '''
  ss = 0.01 #stepsize for x_values in [0,1]
  x_vals = np.arange(0,1+ss,ss)
  f_vals = [f_mu(x, alpha, gamma, sigma, threshold) for x in x_vals]
  plt.plot(x_vals, x_vals, label = 'y=x', color = 'blue') # y = x in blue
  plt.plot(x_vals,f_vals, label = 'f(x)', color = 'red') # f_mu(x) in red
  plt.axvline(x = threshold/(1-alpha), color = 'yellow') # inflection point vertical line in yellow

  x_star1,x_star2 = get_extrema_points(alpha, gamma, sigma, threshold)
  x_1, x_2, x_3 = get_fixedpoints(alpha, gamma, sigma, threshold)

  plt.axvline(x = x_1, color = 'green') # 3 fixed points in green
  plt.axvline(x = x_2, color = 'green')
  plt.axvline(x = x_3, color = 'green')
  plt.axvline(x = x_star1, color = 'black') # first extrema of g(x) = x-f(x)
  plt.axvline(x = x_star2, color = 'black') # second extrema
  plt.grid()
  plt.show()