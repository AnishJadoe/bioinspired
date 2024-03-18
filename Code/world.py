import pygame 

WHITE = (255,255,255)
BLACK = (0,0,0)

class World:
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
        
    def draw_H_world(self):
        self.map.fill((255,255,255))
        background = pygame.image.load("Maps/H_Map.png")
        self.map.blit(background,(0,0))
        return
        
        
    
if __name__ == "__main__":
    import time
    pygame.init()
    
    t_start = time.time()
    t = time.time()
    print("--------BEGIN---------")
    dims = (1400, 788)
    world = World(dims)
    world.draw_H_world()
    pygame.display.update()
    while (t - t_start < 2):
        t = time.time()
    print("--------END---------")
    pygame.quit()

        
