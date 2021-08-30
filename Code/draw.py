import numpy as np
import pygame


class Draw:
    
    """ Class used for drawing
    """    
    
    def __init__(self, dimensions):
        # colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.red = (255, 0, 0)
        self.blue = (0, 0, 255)
        self.yellow = (255, 255, 0)

        self.font = pygame.font.Font('freesansbold.ttf', 50)
        self.text = self.font.render('default', True, self.white, self.black)
        self.textRect = self.text.get_rect()
        self.textRect.center = (dimensions[0] - 600, dimensions[1] - 100)
        self.trail_set = []

    def write_info(self, map, gen, time):
        txt = (f"Generation: {gen}, Time: {time}")
        self.text = self.font.render(txt, True, self.white, self.black)
        map.blit(self.text, self.textRect)