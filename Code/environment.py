import pygame 
import numpy as np

class Envir:
    """This class contains the actual world in which all objects willl interact 
    """    
    
    def __init__(self, dimensions):

        # map dims
        self.height = dimensions[0]
        self.width = dimensions[1]

        # windows settings
        pygame.display.set_caption('Robot World')
        self.map = pygame.display.set_mode((self.height,
                                            self.width))
