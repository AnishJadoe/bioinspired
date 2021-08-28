import pygame 
import numpy as np

class Envir:
    
    def __init__(self, dimensions):

        # map dims
        self.height = dimensions[0]
        self.width = dimensions[1]

        # windows settings
        pygame.display.set_caption('Robot World')
        self.map = pygame.display.set_mode((self.height,
                                            self.width))

    def trail(self, pos):
        for i in range(0, len(self.trail_set) - 1):
            pygame.draw.line(self.map, self.yellow, (self.trail_set[i][0], self.trail_set[i][1]),
                             (self.trail_set[i + 1][0], self.trail_set[i + 1][1]))
        if self.trail_set.__sizeof__() > 3000:
            self.trail_set.pop(0)
        self.trail_set.append(pos)