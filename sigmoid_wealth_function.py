from scipy.stats import norm
import numpy as np
from scipy import optimize

#Maybe make a new class for sigmoid function? (design choice not changing for now)

def f_mu(x, alpha, gamma, sigma, threshold):
  '''
  nu, the mean for Talent is set to zero
  x is the input,(mean wealth in current round)
  S = T + W
  O = alpha T + (1-alpha)W
  See overleaf section 2 (on page 2) for f(x) expression
  The argument to norm cdf is K * (threshold - (1-alpha)*x), where K is positive value
  '''
  return 1 - norm.cdf((threshold - (1-alpha)*x) * np.sqrt((gamma**2 + sigma**2)) / (alpha * gamma**2 + (1-alpha) * sigma**2))

def g(x, alpha, gamma, sigma, threshold):
  '''
  this just returns x-f(x), recall this is positive in (x_1,x_2)
  '''
  return x - f_mu(x, alpha, gamma, sigma, threshold)

def inverse_vals_pdf(y):  
  '''
  Get x such that phi(x) = y
  phi is pdf of standard normal,
  y is some probability value in [0,1]
      returns only the positive x, the other is -1*x
  pdf of standard gaussian e^(-x^2/2)/sqrt(2pi) = y
  '''
  z = y*np.sqrt(2*np.pi)
  x = np.sqrt(-2 * np.log(z))
  return x

def get_extrema_points(alpha, gamma, sigma, threshold):
  '''
  get extrema points for g(x) = x-f_mu(x,alpha, gamma, sigma, threshold)
  i.e where f'(x) = 1
  i.e K*(1-alpha)* gaussian_pdf((threshold-(1-alpha)x)*K) = 1
  
  returns (x_ext1,x_ext2) where 0 <= x_ext1 <= x_ext2 <= 1
  '''
  K = np.sqrt((gamma**2 + sigma**2)) / (alpha * gamma**2 + (1-alpha) * sigma**2)
  inv_term = inverse_vals_pdf(1 / (K*(1-alpha)))
  return (threshold - inv_term/K)/(1-alpha), (threshold + inv_term/K)/(1-alpha)

def get_fixedpoints(alpha, gamma, sigma, threshold):
  '''
  Say x_star1, x_star2 are the extrema for g(x), and g(x_star1) > 0 and g(x_star2) < 0 
  so we know one root for g(x) is in (0,x_star1)
             2nd root for g(x) is in (x_star1, x_star2)
             3rd root for g(x) is in (x_star2, 1)
  the 3 return values fp fp1<fp2<fp3
  '''
  x_star1, x_star2 = get_extrema_points(alpha, gamma, sigma, threshold)
  fp1 = optimize.brentq(g, 0, x_star1, args = (alpha,gamma,sigma,threshold))
  fp2 = optimize.brentq(g, x_star1, x_star2, args = (alpha,gamma,sigma,threshold))
  fp3 = optimize.brentq(g, x_star2, 1, args = (alpha,gamma,sigma,threshold))
  return fp1, fp2, fp3

def get_Delta(alpha, gamma, sigma, threshold):
  x_star1, x_star2 = get_extrema_points(alpha, gamma, sigma, threshold)
  return g(x_star1, alpha, gamma, sigma, threshold)

def inflection_point_greater_than1(threshold, alpha):
  '''
  return True if inflection point i.e threshold/(1-alpha) >= 1.0
  This implies only 1 fixed point
  '''
  return threshold / (1-alpha) >= 1.0

def threefixedpoints(alpha, gamma, sigma, threshold):
  '''
  Return true only if 3 fixed points
  '''
  if inflection_point_greater_than1(threshold, alpha):
    return False
  x_star1, x_star2 = get_extrema_points(alpha, gamma, sigma, threshold)
  if 0 <= x_star1 <= x_star2 <= 1:
    if g(x_star1, alpha, gamma, sigma, threshold) > 0 and g(x_star2, alpha, gamma, sigma, threshold) < 0:
      return True
  return False