import pygame.locals

from src.controllers.neural_net import NeuralNet
from src.utility.constants import *
import numpy as np
MAX_SPEED = 128
MIN_SPEED = -128
M2P = 3779.52

class BaseController():
    def __init__(self):
        self.name = "BaseController"
        
    def calculate_motor_speed(self):
        pass
    
    def scale_action(self, action):
        """
        Scales the action output from the neural network to the motor speed range.

        Parameters:
            action (float): The action output from the neural network.

        Returns:
            float: The scaled motor speed.
        """
        return (action + 0.5) * (MAX_SPEED - MIN_SPEED) + MIN_SPEED
        
class ManualController(BaseController):
    def __init__(self):
        super().__init__()
        self.name = "Manual Controller"
    
    def calculate_motor_speed(self, event, vl,vr):
        
        if event.type in {pygame.KEYDOWN}:
            if event.key == pygame.locals.K_a:
                vl = min(MAX_SPEED, vl+25)
            elif event.key == pygame.locals.K_s:
                vl = max(MIN_SPEED, vl-25)
            elif event.key == pygame.locals.K_k:
                vr = min(MAX_SPEED, vr + 25)
            elif event.key == pygame.locals.K_j:
                vr = max(MIN_SPEED, vr - 25)
                
        return (vl,vr)

                
class NeuroController(BaseController):
    def __init__(self,chromosome,n_inputs=N_INPUTS,n_outputs=N_OUTPUTS, n_hidden=N_HIDDEN):
        super().__init__()
        self.name = "Neuro Controller"
        self.network = NeuralNet(n_inputs=n_inputs,n_outputs=n_outputs,n_hidden=n_hidden,chromosome=chromosome)
    
    def calculate_motor_speed(self, state):
        motor_speeds = self.network.forward_pass(state)
        vl = np.clip(self.scale_action(motor_speeds[0, 0]), MIN_SPEED, MAX_SPEED)
        vr = np.clip(self.scale_action(motor_speeds[1, 0]), MIN_SPEED, MAX_SPEED)
        
        return float(vl), float(vr)
        