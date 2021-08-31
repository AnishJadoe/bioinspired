'''
By running this file, the whole algorithm is run and plots are made of the given paremeters 
'''
# Loading in functions
from plot import plot_results


epochs = 10
time = 30000
ls_cross_rate = [0.8]
ls_mut_rate = [1]

plot_results(epochs = epochs,time = time,ls_cross = ls_cross_rate,ls_mut = ls_mut_rate, run=0,)
