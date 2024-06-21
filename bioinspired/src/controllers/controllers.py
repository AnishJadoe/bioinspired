import pygame.locals

from src.controllers.neural_net import NeuralNet

MAX_SPEED = 255
MIN_SPEED = -255
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
        
        if event.type in {pygame.KEYDOWN, pygame.KEYUP}:
            if event.key == pygame.locals.K_a:
                vl += 0.01 * M2P
            elif event.key == pygame.locals.K_s:
                vl -= 0.01 * M2P
            elif event.key == pygame.locals.K_k:
                vr += 0.01 * M2P
            elif event.key == pygame.locals.K_j:
                vr -= 0.01 * M2P
                
        return (vl,vr)

class NeuroController(BaseController):
    def __init__(self):
        super().__init__()
        self.name = "Neuro Controller"
        self.network = NeuralNet()
    
    def calculate_motor_speed(self, state):
        return self.network.forward_pass(state)
        