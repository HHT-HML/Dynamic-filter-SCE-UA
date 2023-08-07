import numpy as np
import logging
import os
import sys
import time
from SALib.analyze import morris
from SALib.test_functions import Sobol_G
from SALib.plotting.morris import (
horizontal_bar_plot,
covariance_plot
)
import matplotlib as plt
# Read the parameter range file and generate samples
# problem = read_param_file("./src/SALib/test_functions/params/Sobol_G.txt")
bl = np.array([-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32])
bu = np.array([32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32])
x0 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
# or define manually without a parameter file:
problem = {
   'num_vars': 20,
   'names': ['x1', 'x2', 'x3', 'x4','x5','x6', 'x7', 'x8', 'x9','x10','x11', 'x12', 'x13', 'x14','x15','x16', 'x17', 'x18', 'x19','x20'],
   'groups': None,
   'bounds': [[-32, 32],
             [-32, 32],
             [-32, 32],
             [-32, 32],
              [-32, 32],
[-32, 32],
             [-32, 32],
             [-32, 32],
             [-32, 32],
              [-32, 32],
[-32, 32],
             [-32, 32],
             [-32, 32],
             [-32, 32],
              [-32, 32],
[-32, 32],
             [-32, 32],
             [-32, 32],
             [-32, 32],
              [-32, 32],
              ]}
# Files with a 4th column for "group name" will be detected automatically, e.g.
# param_file = '../../src/SALib/test_functions/params/Ishigami_groups.txt'

param_values = []
# To use optimized trajectories (brute force method),
# give an integer value for optimal_trajectories

# Run the "model" -- this will happen offline for external models
# Y = Sobol_G.evaluate(param_values)
# 初始数组

Y = Sobol_G.evaluate(param_values,bl,bu,x0,problem)
# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
while len(Y)%(problem["num_vars"]+1)!=0:
    Y.append(Y[len(Y)-1])
    param_values.append(param_values[len(param_values)-1])
Y=np.array(Y)
param_values=np.array(param_values)
Si = morris.analyze(
    problem,
    param_values,
    Y,
    conf_level=0.95,
    print_to_console=True,
    num_levels=4,
    num_resamples=100,
)
# Returns a dictionary with keys 'mu', 'mu_star', 'sigma', and 'mu_star_conf'
# e.g. Si['mu_star'] contains the mu* value for each parameter, in the
# same order as the parameter file
si_path=os.getcwd()+os.sep+"Si_final.txt"
si_file=open(si_path,"w+")
for i in range(len(Si["mu"])):
    si_file.write(str(Si["mu"][i])+str('    ')+str(Si["mu_star"][i])+'\n')
si_file.write(str("-----------------------------------------")+'\n')
si_file.close()
fig, (ax1, ax2) = plt.subplots(1, 2)
horizontal_bar_plot(ax1, Si, {}, sortby="mu_star", unit=r"tCO$_2$/year")
covariance_plot(ax2, Si, {}, unit=r"tCO$_2$/year")

#fig2 = plt.figure()
#sample_histograms(fig2, param_values, problem, {"color": "y"})
plt.show()
