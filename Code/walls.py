import numpy as np
import pygame


BLACK = (0, 0, 0, 255)
YELLOW = (255, 255, 0, 255)


class Map:  # change name to map
    def __init__(self, world: pygame.surface, cell_size):
        self.world = world
        self.world_width, self.world_height = world.get_size()
        self.size = cell_size
        self.amount_tokens = 10

        self.world_data = self.get_world_data()
        self.grid = self.get_grid()
        self.move_cells = self.get_move_cells()
        self.walls = self.get_walls()
        self.tokens = self.get_tokens()

    def get_world_data(self):
        return np.array(
            [
                [
                    int(self.world.get_at((x, y)) == BLACK)
                    for x in range(0, self.world_width, self.size)
                ]
                for y in range(0, self.world_height, self.size)
            ]
        ).T

    def get_maze_map(self):
        WALL = 0
        FLOOR = 1

        np.random.seed(42)
        cellMAP = np.random.choice(
            [1, 0],
            size=(
                int(self.world_height / self.size),
                int(self.world_width / self.size),
            ),
            p=[0.60, 0.40],
        )

        generations = 2
        GRIDHEIGHT = int(self.world_height / self.size)
        GRIDWIDTH = int(self.world_width / self.size)

        if generations > 1:
            for generation in range(generations):
                for row in range(GRIDHEIGHT):
                    for column in range(GRIDWIDTH):
                        subMAP = cellMAP[
                            max(row - 1, 0) : min(row + 2, GRIDHEIGHT),
                            max(column - 1, 0) : min(column + 2, GRIDWIDTH),
                        ]

                        wallcount_1away = len(np.where(subMAP.flatten() == WALL)[0])

                        subMAP = cellMAP[
                            max(row - 2, 0) : min(row + 3, GRIDHEIGHT),
                            max(column - 2, 0) : min(column + 3, GRIDWIDTH),
                        ]
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

                            if (
                                row == 0
                                or column == 0
                                or row == GRIDHEIGHT - self.size
                                or column == GRIDWIDTH - self.size
                            ):
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
        """
        Draw the black rectangles based on the cellMAP obtained from the
        game of life.
        :param map:
        :return:
        """

        grid = []

        for x in range(0, self.world_height, self.size):
            for y in range(0, self.world_width, self.size):
                rect = pygame.Rect(x, y, self.size, self.size)
                grid.append(rect)

        return grid

    def get_walls(self):
        walls = []
        for cell in self.grid:
            x = int(cell.y / self.size)  # Check middle of the cell
            y = int(cell.x / self.size)
            if self.world_data[x][y] == 1:
                copy_cell = cell.copy()
                rotate_x = cell.y
                rotate_y = cell.x
                copy_cell.x = rotate_x
                copy_cell.y = rotate_y
                walls.append(copy_cell)

        return walls

    def get_move_cells(self):
        move_cells = []

        for cell in self.grid:
            x = int(cell.y / self.size)
            y = int(cell.x / self.size)
            if self.world_data[x][y] == 0:
                copy_cell = cell.copy()
                rotate_x = cell.y
                rotate_y = cell.x
                copy_cell.x = rotate_x
                copy_cell.y = rotate_y
                move_cells.append(copy_cell)

        return move_cells

    def get_tokens(self):
        tokens = []
        for i in range(self.amount_tokens):
            token_idx = np.random.randint(low=0, high=len(self.move_cells))
            tokens.append(self.move_cells[token_idx])

        return tokens

    def draw_walls(self, map):
        for ob in self.walls:
            pygame.draw.rect(map, BLACK, ob)

    def update_tokens(self, map, robots):
        for token in self.tokens:
            pygame.draw.rect(map, YELLOW, token, 0)
            for robot in robots:
                if token.colliderect(robot.rect):
                    self.tokens.pop(self.tokens.index(token))
                    robot.token += 1
                    break
