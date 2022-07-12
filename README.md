# Wealth Feedback Loops
Optimal Interventions in Wealth Feedback Loops

# Workflow

To explore the results open the jupyter notebook ''process_savedresult.ipynb''

To change the grid spacing and granularity go into run_andsave.py, change those spacings and then run ''python run_andsave.py''
This will also save to a numpy output file, i.e. the results_array is dumped using ''numpy.save(,)''

The jupyter notebook can process this saved file and validate our hypothesis empirically 

# About the scripts

Paramclass is a class which is used to create an object that has member variables alpha, gamma, sigma, threshold, lambda and discount factor

alpha, gamma, sigma, threshold uniquely give a wealth sigmoid

lambda and discount factor are required for the total objective computation

Basically for each alpha, gamma, sigma, threshold, lambda, discount factor in the parameter grid we have the corresponding paramclass object appended to a results_array