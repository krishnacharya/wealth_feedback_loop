from grid_search import *
savefilename = "0.1grid_granu100"

#alpha_step,gamma_step,...lambda_step ,discountfactor_step are all used to create the discrete grid for the grid search
#granularity defines how many cost values we want to look through, for e.g granu = 10, means stepsize = (x_2 -x_1 -Delta) / 10
grid_dump(savefilename, alpha_step=0.1, gammma_step=0.1, sigma_step=0.1, threshold_step=0.1, lamb_step=0.1, df_step=0.1, granu=100)