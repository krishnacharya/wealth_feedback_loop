from helpers import *
from objective import *
from sigmoid_wealth_function import *

class Cost_Tags:
  '''
  This class's object stores the total objective, 
  total rounds, percent overshoot beyond x_2 etc .. for a specific cost intervention
  
  This class's object is used as a member variable in Paramclass (see set_obj(self) method in Paramclass!)
  Basically Paramclass has a List of costs(from Delta to x_2-x_1) for the unique tuple (alpha, gamma, sigma, threshold)

  For this each element in this List of costs we have a corresponding total_objective (also depends on lambda and gamma)
  In addition I also store useful information like % of overshoot beyond x_2, total_rounds to reach x_2, etc.
  '''
  def __init__(self, cost):
    self.cost = cost
    self.total_objective = 0
    self.total_rounds = 0
    self.end_wealth = 0

class Paramclass:
  def __init__(self, alpha, gamma, sigma, threshold, lamb, df):
    #sigmoid parameters, maybe inherit from a parent class(design to change later)
    self.alpha = alpha
    self.gamma = gamma #dont confuse this with discount factor used for objective
    self.sigma = sigma
    self.threshold = threshold

    #objective parameters
    self.lamb = lamb #lambda, i.e the weight to the cost term, 1-lambda is the weight to reward term
    self.df = df #discount factor

    #some theoretical sufficient conditions which may not work empirically(since we need not exactly reach x_2)
    self.suff1_oneshotbest = True if self.lamb <= 0.5 else False #Claim 25 in overleaf, if you weight distance to x_2 more
    self.suff2_oneshotbest = True if self.df >= (2 - 1/self.lamb) else False #see Theorem2 in overleaf

    #will be calculated and set later
    self.cost_vec = None #interventions go from Delta+tol to x_2-x_1 in uniform spacing
    self.obj_vec = None

    # self.oneshot_isbest = False
    # self.slow_isbest = False
    # self.monotonic = False

  def set_FP_Delta(self, x_1, x_2, x_3, Delta):
    '''
    set fixed points and Delta(max x-f(x)) values
    '''
    self.x_1 = x_1
    self.x_2 = x_2
    self.x_3 = x_3
    self.Delta = Delta

  def set_cost_vec(self, cost_vec):
    self.cost_vec = cost_vec

  def constant_intervention(self, cost_obj):
    '''
    x(t+1) = f(x(t)+c), stop when x(t+1) >= x_2
    c is the constant cost intervention, always pass a value > Delta, I dont explicity check this
    Technically not ending exactly at x_2
    sets total_objective, total time steps, final wealth, how much percent its above x_2
    '''
    x_c = self.x_1 #current wealth
    t = 1
    while(x_c < self.x_2):
      x_next = f_mu(x_c + cost_obj.cost, self.alpha, self.gamma, self.sigma, self.threshold) #x(t+1) = f(x(t) + c)
      dist_to_x2 = max(0, self.x_2 - x_next) # just to ensure beyond x_2, dist to x_2 is zero
      cost_obj.total_objective += (self.df**(t-1)) * (self.lamb*cost_obj.cost + (1 - self.lamb)*dist_to_x2)
      t += 1
      x_c = x_next
    cost_obj.total_rounds = t-1
    cost_obj.end_wealth = x_c
    cost_obj.overshoot_percent = 100*((x_c-self.x_2)/self.x_2)

  def set_obj(self):
    '''
    For each c in self.costvec we want the corresponding objective
    '''
    self.obj_vec = []#each cost in cost_vec corresponds to one CostTag object
    for cost in self.cost_vec:
      cost_obj = Cost_Tags(cost)
      self.constant_intervention(cost_obj)
      self.obj_vec.append(cost_obj)
  # def set_monotonicity(self):
  #   '''
  #   set monotonicity only after self.obj_vectors are set
  #   Attributes for the objective vector
  #   '''
  #   if non_increasing(self.obj_vec):
  #     self.non_increasing = True
  #     self.monotonic = True
  #     self.oneshot_isbest = True
  #   if non_decreasing(self.obj_vec):
  #     self.non_decreasing = True
  #     self.monotonic = True
  #     self.slow_isbest = True
    


