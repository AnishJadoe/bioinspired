import pygame 
import numpy as np
from ..utility.constants import *


class Tile():
    def __init__(self, tile, tile_id):
        self.hitbox = tile
        self.x = tile.x
        self.y = tile.y
        self.id = tile_id
        
class Target():
    def __init__(self, target, tile_id):
        self.hitbox = target
        self.being_carried = False
        self.x = target.x
        self.y = target.y
        self.id = tile_id
    
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
        tile_id = 0
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
                    wall = Tile(obj, tile_id)
                    self._draw_wall(wall)
                    self.walls.append(wall)
                    self.wall_map[x][y] = 1
                    self._add_obstacle(wall)
                if char == "T":
                    target = Target(obj, tile_id)
                    self._draw_token(target)
                    self.tokens.append(target)
                    self.token_map[x][y] = 1
                    self._add_token(target)
                if char == "-":
                    move_tile = Tile(obj, tile_id)
                    self._draw_move_tiles(move_tile)
                    self.movable_tiles.append(move_tile)
                    self._add_movable_tile(move_tile)
                if char == "S":
                    start_pos = Tile(obj, tile_id)
                    self._draw_start_position(start_pos)
                    self._add_start_tile(start_pos)
                    self.start_pos = start_pos
                if char == "E":
                    end_pos = Tile(obj, tile_id)
                    self._draw_end_position(end_pos)
                    self.end_pos = end_pos
                tile_id += 1
        return
            
    def _draw_wall(self, obj: Tile):
        pygame.draw.rect(self.surf, BLACK, obj.hitbox)
    
    def _draw_token(self, obj: Target):
        if obj.being_carried:
            obj.hitbox.topleft = (obj.x,obj.y)
            pygame.draw.rect(self.surf, DARK_GREEN, obj.hitbox)
        else:
            pygame.draw.rect(self.surf, YELLOW, obj.hitbox)

    def _draw_move_tiles(self, obj: Tile):
        pygame.draw.rect(self.surf, WHITE, obj.hitbox)

    def _draw_start_position(self, obj: pygame.Rect):
        pygame.draw.rect(self.surf, RED, obj.hitbox)
    
    def _draw_end_position(self, obj: Tile):
        pygame.draw.rect(self.surf, ORANGE, obj.hitbox)
    
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
        
    
    