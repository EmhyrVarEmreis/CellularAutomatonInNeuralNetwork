from CellState import CellState


class World:
    width = 0
    height = 0
    world = []
    worldCopy = []

    def __init__(self, width, height):
        self.width = width
        self.height = height
        for i in range(0, self.height):
            new = []
            for j in range(0, self.width):
                new.append(CellState.Dead)
            self.world.append(new)

    def get_cell(self, i, j):
        if i < 0 or j < 0 or i >= self.width or j >= self.height:
            return CellState.Dead
        else:
            return self.world[i][j]

    def set_in_world_copy(self, i, j, state):
        self.worldCopy[i][j] = state

    def set_in_world(self, i, j, state):
        self.world[i][j] = state

    def prepare_copy(self):
        for i in range(0, self.height):
            new = []
            for j in range(0, self.width):
                new.append(CellState.Dead)
            self.worldCopy.append(new)

    def switch_copy(self):
        self.world = self.worldCopy
        self.worldCopy = []

    def print(self):
        print('\n'.join([''.join(['{:3}'.format(item) for item in row]) for row in self.world]))
