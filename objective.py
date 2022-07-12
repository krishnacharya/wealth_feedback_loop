import numpy as np
from sigmoid_wealth_function import *
# def one_shot_obj(x_1, x_2, lamb = 0.5):
#   '''
#     cost is lambda(c + (x_2 - x(t+1))), but x(2) = x_2, and c = x_2-x_1
#   '''
#   return lamb*(x_2-x_1)

# def constant_obj(x_1, x_2, f_args, c, discount_factor=1, lamb = 0.5):
#   '''
#   x(t+1) = f(x(t)+c), stop when x(t+1) >= x_2
#   c is the constant cost intervention, always pass a value > Delta, I dont explicity check this
#   Technically not ending exactly at x_2
#   returns total_cost, total time steps, final wealth, how much percent its above x_2
#   '''
#   x_c = x_1 #current wealth
#   t = 1
#   total_objective = 0.0
#   status = "exited within 10000 steps"
#   while(x_c < x_2):
#         x_next = f_mu(x_c + c, f_args[0], f_args[1], f_args[2], f_args[3]) #x(t+1) = f(x(t) + c)
#         reward = max(0, x_2 - x_next) # just to ensure beyond x_2 rewards are zero
#         total_objective = total_objective + (discount_factor**(t-1)) * (lamb*c + (1-lamb)*reward)
#         t += 1
#         x_c = x_next
#   return status, total_objective, t-1, x_c, str(100*((x_c-x_2)/x_2)) + " % Overshoot"

def get_cost_vec(x_1, x_2, Delta, granu = 10, tol = 1e-4): #make granularity tunable
  ss = ((x_2-x_1) - Delta) / granu # effectively want 'granu' number of c values from x_2-x_1 to Delta
  cost_vec = np.arange(Delta + tol, x_2-x_1, ss)
  if x_2-x_1 not in cost_vec:
    cost_vec = np.append(cost_vec, x_2-x_1)
  return cost_vec

# def get_objvector_for_interventions(x_1, x_2, f_args, Delta, granu = 100, df = 1.0, lamb = 0.5, tol = 1e-4):
#   '''
#   intervention value ranges from Delta to x_2-x_1 (uniform grid)
#   granu is the number of c values to pick (the grids granulaity, higher value means finer)
#   df is discount factor
#   lambd is weight to cost
#   '''
#   ss = ((x_2-x_1) - Delta)/granu # effectively want 'granu' number of c values from x_2-x_1 to Delta
#   c_vals = np.arange(Delta + tol, x_2-x_1, ss)
#   c_vals = np.append(c_vals, x_2-x_1)#change to append only if x_2-x_1 not there
#   return c_vals, np.array([constant_obj(x_1, x_2, f_args, c, discount_factor = df, lamb = lamb)[1] for c in c_vals])
