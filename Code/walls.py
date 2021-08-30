import numpy as np
import pygame 

class Walls:  # change name to world

    def __init__(self, size, dimensions):
        self.black = (0, 0, 0)
        self.yellow = (255, 255, 0)

        self.size = size
        self.width = dimensions[1]
        self.height = dimensions[0]
        self.amount_rewards = 20

        self.cellMAP = self.get_maze_map()
        self.grid = self.get_grid()
        self.obstacles = self.get_obs()
        self.rewards = list()

    def get_maze_map(self):

        WALL = 0
        FLOOR = 1

        np.random.seed(42)
        cellMAP = np.random.choice([1, 0], size=(int(self.height / self.size), int(self.width / self.size)),
                                   p=[0.80, 0.20])

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

                        # this consolidates walls
                        # for first five generations build a scaffolding of wal

                        if generation < generations - 1:
                            # if looking 1 away in all directions you see 5 or more walls
                            # consolidate this point into a wall, if that doesnt happpen
                            # and if looking 2 away in all directions you see less than
                            # 7 walls, add a wall, this consolidates and adds walls
                            if wallcount_1away >= 5 or wallcount_2away <= 7:
                                cellMAP[row][column] = WALL

                            else:
                                cellMAP[row][column] = FLOOR

                            if row == 0 or column == 0 or row == GRIDHEIGHT - self.size \
                                    or column == GRIDWIDTH - self.size:
                                cellMAP[row][column] = WALL
                        # this consolidates open space, fills in standalone walls,
                        # after generation 5 consolidate walls and increase walking space
                        # if there are more than 5 walls nearby make that point a wall,
                        # otherwise add a floor
                        else:
                            if wallcount_1away >= 5:
                                cellMAP[row][column] = WALL
                            else:
                                cellMAP[row][column] = FLOOR

        return cellMAP

    def get_grid(self):
        '''
        Draw the black rectangles based on the cellMAP obtained from the
        game of life.
        :param map:
        :return:
        '''

        grid = []

        for x in range(0, self.height, self.size):
            for y in range(0, self.width, self.size):
                rect = pygame.Rect(x, y, self.size, self.size)
                grid.append(rect)

        return grid

    def get_obs(self):

        obs = []

        for index in range(0, len(self.grid)):
            if self.cellMAP.flatten()[index] == 0:
                obs.append(self.grid[index])

        return obs

    def get_rewards(self):

        ls_coors = []

        for index in range(0, len(self.grid)):
            if self.cellMAP.flatten()[index] == 1:
                ls_coors.append(self.grid[index])

        for i in range(0, self.amount_rewards):
            reward_idx = np.random.randint(low=0, high=len(ls_coors))
            if len(self.rewards) < 20:
                self.rewards.append(ls_coors[reward_idx])

    def draw(self, map, robots):

        for ob in self.obstacles:
            pygame.draw.rect(map, self.black, ob, 0)

        for reward in self.rewards:
            pygame.draw.rect(map, self.yellow, reward, 0)

            for robot in robots:
                if reward.colliderect(robot.rect):
                    self.rewards.pop(self.rewards.index(reward))
                    break