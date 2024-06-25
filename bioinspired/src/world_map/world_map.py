import pygame 
import numpy as np
from ..utility.constants import *


class WorldMap:
    def __init__(self, skeleton_file):
        
        self.skeleton_file = skeleton_file
        self.map_width, self.map_height = self._get_map_dims()
        self.tile_size = CELL_SIZE
        self.surf = []  # World map surface to draw on
        self.wall_map = np.zeros((self.map_width,
                            self.map_height))
        self.token_map = np.zeros((self.map_width,
                            self.map_height))
        self.end_pos = None
        # Save for future reference 
        self.walls = []
        self.tokens = []
        self.start_pos = (0,0)
        self.movable_tiles = []
        self.spatial_grid = [[[] for _ in range(self.map_height)] for _ in range(self.map_width)]
    
    def _get_map_dims(self):
        height = 0
        width = 0
        skeleton_file = self._load_skeleton_file(self.skeleton_file)
        for line in skeleton_file:
            height += 1
        width = len(line)
        return width, height
    
        
    def build_map(self):
        self.surf = pygame.display.set_mode(size=(
            self.map_width*self.tile_size,
            self.map_height*self.tile_size),
            display=0)
        self._parse_skeleton_file()

    def clear_map(self):
        # Clear all variables
        self.surf = []  
        self.wall_map = np.zeros((self.map_width,
                            self.map_height))
        self.token_map = np.zeros((self.map_width,
                    self.map_height))
        self.walls = []
        self.tokens = []
        self.movable_tiles = []
        self.spatial_grid = [[[] for _ in range(self.map_height)] for _ in range(self.map_width)]
    
    def _add_obstacle(self, obstacle):
        x, y = obstacle.x // self.tile_size, obstacle.y // self.tile_size
        self.spatial_grid[int(x)][int(y)].append(obstacle)

    def _add_token(self,token):
        x, y = token.x // self.tile_size, token.y // self.tile_size
        self.spatial_grid[int(x)][int(y)].append(token)
        
    def _add_movable_tile(self,move_tile):
        x, y = move_tile.x // self.tile_size, move_tile.y // self.tile_size
        self.spatial_grid[int(x)][int(y)].append(move_tile)
        
    def _add_start_tile(self,start_tile):
        x, y = start_tile.x // self.tile_size, start_tile.y // self.tile_size
        self.spatial_grid[int(x)][int(y)].append(start_tile)
        
    def _load_skeleton_file(self, file):
        return open(file, "r")
    
    def _parse_skeleton_file(self):
        skeleton_file = self._load_skeleton_file(self.skeleton_file)
        for y, line in enumerate(skeleton_file):
            for x, char in enumerate(line):
                obj = pygame.Rect(
                    x*self.tile_size,
                    y*self.tile_size,
                    self.tile_size,
                    self.tile_size
                    )
                if char == "*":
                    self._draw_wall(obj)
                    self.walls.append(obj)
                    self.wall_map[x][y] = 1
                    self._add_obstacle(obj)
                if char == "T":
                    self._draw_token(obj)
                    self.tokens.append(obj)
                    self.token_map[x][y] = 1
                    self._add_token(obj)
                if char == "-":
                    self._draw_move_tiles(obj)
                    self.movable_tiles.append(obj)
                    self._add_movable_tile(obj)
                if char == "S":
                    self._draw_start_position(obj)
                    self.start_pos = obj
                    self._add_start_tile(obj)
                if char == "E":
                    self._draw_end_position(obj)
                    self.end_pos = obj
        return
            
    def _draw_wall(self, obj):
        pygame.draw.rect(self.surf, BLACK, obj)
    
    def _draw_token(self, obj):
        pygame.draw.rect(self.surf, YELLOW, obj)
      
    def _draw_move_tiles(self, obj):
        pygame.draw.rect(self.surf, WHITE, obj)

    def _draw_start_position(self, obj: pygame.Rect):
        pygame.draw.rect(self.surf, RED, obj)
    
    def _draw_end_position(self, obj):
        pygame.draw.rect(self.surf, ORANGE, obj)
    
    def update_map(self):

        self.surf.fill(WHITE)
        self._draw_start_position(self.start_pos)
        if self.end_pos:
            self._draw_end_position(self.end_pos)
        for wall in self.walls:
            self._draw_wall(wall)
        for token in self.tokens:
            self._draw_token(token)
        for moveable_tile in self.movable_tiles:
            self._draw_move_tiles(moveable_tile)
        
        return
        

if __name__ == "__main__":
    pygame.init()
    pygame.display.set_caption('Robot World')
    map = WorldMap(skeleton_file="Maps/H_map.txt", 
                map_width=60,
                map_height=40,
                tile_size=20)
    map.build_map()
    
    clock = pygame.time.Clock()
    time = 30000
    while pygame.time.get_ticks() <= time:
        clock.tick(20)
        pygame.display.update()
        
    
    