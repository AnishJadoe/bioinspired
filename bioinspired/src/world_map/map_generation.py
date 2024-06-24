import random
import numpy as np
from ..utility.constants import *

class MapGenerator:
    def __init__(self, width=60, height=40, start_pos=(0, 0), num_tokens=10, size=1):
        self.width = width
        self.height = height
        self.start_pos = start_pos
        self.num_tokens = num_tokens
        self.size = size
        self.map = [['-' for _ in range(width)] for _ in range(height)]

    def get_maze_map(self):
        WALL = 0
        FLOOR = 1

        np.random.seed(42)
        cellMAP = np.random.choice([FLOOR, WALL],
                                   size=(int(self.height / self.size), int(self.width / self.size)),
                                   p=[0.60, 0.40])

        generations = 2
        GRIDHEIGHT = int(self.height / self.size)
        GRIDWIDTH = int(self.width / self.size)

        if generations > 1:
            for generation in range(generations):
                for row in range(GRIDHEIGHT):
                    for column in range(GRIDWIDTH):
                        subMAP = cellMAP[max(row - 1, 0):min(row + 2, GRIDHEIGHT),
                                         max(column - 1, 0):min(column + 2, GRIDWIDTH)]

                        wallcount_1away = len(np.where(subMAP.flatten() == WALL)[0])

                        subMAP = cellMAP[max(row - 2, 0):min(row + 3, GRIDHEIGHT),
                                         max(column - 2, 0):min(column + 3, GRIDWIDTH)]
                        wallcount_2away = len(np.where(subMAP.flatten() == WALL)[0])

                        if generation < generations - 1:
                            if wallcount_1away >= 5 or wallcount_2away <= 7:
                                cellMAP[row][column] = WALL
                            else:
                                cellMAP[row][column] = FLOOR

                            if row == 0 or column == 0 or row == GRIDHEIGHT - self.size \
                                    or column == GRIDWIDTH - self.size:
                                cellMAP[row][column] = WALL
                        else:
                            if wallcount_1away >= 5:
                                cellMAP[row][column] = WALL
                            else:
                                cellMAP[row][column] = FLOOR

        return cellMAP

    def initialize_map(self):
        cellMAP = self.get_maze_map()
        for y in range(len(cellMAP)):
            for x in range(len(cellMAP[y])):
                self.map[y][x] = '*' if cellMAP[y][x] == 0 else '-'

        # Set the borders to walls
        for x in range(self.width):
            self.map[0][x] = '*'
            self.map[self.height - 1][x] = '*'
        for y in range(self.height):
            self.map[y][0] = '*'
            self.map[y][self.width - 1] = '*'

    def place_tokens(self):
        for _ in range(self.num_tokens):
            while True:
                x = random.randint(1, self.width - 2)
                y = random.randint(1, self.height - 2)
                if self.map[y][x] == '-':
                    self.map[y][x] = 'T'
                    break

    def set_start_position(self):
        x, y = self.start_pos
        self.map[y][x] = 'S'

    def generate_map(self):
        self.initialize_map()
        self.place_tokens()
        self.set_start_position()

    def save_map_to_file(self, filename="map.txt"):
        with open(filename, 'w') as file:
            for line in self.map:
                file.write("".join(line) + "\n")
                
# Example usage
if __name__ == "__main__":
    width = MAP_DIMS[0]
    height = MAP_DIMS[1]
    start_pos = START_POSITION  # Example start position (x, y)
    num_tokens = N_TOKENS

    map_gen = MapGenerator(width=width, height=height, start_pos=start_pos, num_tokens=num_tokens)
    map_gen.generate_map()
    map_gen.save_map_to_file("bioinspired/src/world_map/maps/random_map.txt")