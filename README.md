# wealth_feedback_loop
Optimal Interventions in Wealth Feedback Loops

To explore the results open the jupyter notebook ''process_savedresult.ipynb''

To change the grid spacing, granulity go into run_andsave.py and change those entries and run ''python run_andsave.py''

# About the scripts

paramclass is a class which is used to create an object which has member variables alpha, gamma, sigma, threshold, lambda and discount factor

alpha, gamma, sigma, threshold uniquely give a wealth sigmoid

lambda and discount factor are required for the total objective computation

Basically for each alpha, gamma, sigma, threshold, lambda, discount factor in the parameter grid we have that paramclass object appended to a resultlist

This resultlist is dumped using ''numpy.save(,)''

The jupyter notebook can process this saved file and validate our hypothesis' empirically 