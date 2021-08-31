# Bioinspired Intelligence 
This is code written for the Bio-Inspired Intelligence course given at the TU Delft. The assignment consisted of developing an evolutionary robotics algorithm for a differential drive robot. The following code consists of an algorithm that generates a map and an algorithm that uses evolutionary principles to develop autonomous control behavior on a robot. 


## Code Overview
- The 'main.py' file runs the complete program and plots the performance of the corresponding setting of the algorithm. 
- The 'robot.py' file contains all the code that was needed to simulate the robot, this includes the code for the kinematics and of the fitness function
- The 'walls.py' file contains the code to generate the walls within the environment. The walls are generated usign an algorithm that resembels Conways game of life
- In both the 'draw.py' and 'enviroment.py' file different functions are created that help with making the overworld in which the robot has to navigate itself
- The 'run_simulation.py' file contains all the code that was needed to run the world in which both the enviroment and the robot have to operate
- The 'genetic_algorithm.py' contains the code that runs all the operators that were described in the report. This includes the crossover, mutation and selection operators
- The 'plot.py' file contains the code that is needed to plot the results that are obtained after the algorithm is run 
- 'main.py' can be run and this should let the algorithm do its thing
