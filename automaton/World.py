import random

from PIL import Image

from automaton.CellState import CellState, cell_state_from


class World:
    width = 0
    height = 0
    world = []
    worldCopy = []

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.clear()

    def clear(self):
        self.world = []
        for i in range(0, self.height):
            new = []
            for j in range(0, self.width):
                new.append(CellState.Dead)
            self.world.append(new)

    def make_random(self, percentage):
        self.clear()
        alive_count = int(self.width * self.height * percentage / 100)

        random.seed(None)

        while alive_count > 0:
            while True:
                a = random.randint(0, self.width - 1)
                b = random.randint(0, self.height - 1)
                if self.world[a][b] == CellState.Dead:
                    break
            self.world[a][b] = CellState.Alive
            alive_count -= 1

    def get_cell(self, i, j):
        if i < 0 or j < 0 or i >= self.width or j >= self.height:
            return self.world[abs(i % self.width)][abs(j % self.height)]
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

    def save(self, file_path):
        with open(file_path, 'w') as f:
            f.write('\n'.join([''.join(['{:1}'.format(item) for item in row]) for row in self.world]))

    def load(self, file_path):
        with open(file_path, 'r') as f:
            self.world = []
            for l in f.readlines():
                new = []
                if (not l.startswith('#')) and (not l.strip() == ''):
                    for i in l:
                        if not i.strip() == '':
                            new.append(cell_state_from(int(str(i).strip())))
                if len(new) > 0:
                    self.world.append(new)

    def save_as_image(self, path, scale=1):
        im = self.get_as_image(scale)
        im.save(path)

    def get_as_image(self, scale=1):
        im = Image.new('RGB', (self.width, self.height))
        im.putdata([(int(x / 1 * 255), int(x / 1 * 255), int(x / 1 * 255)) for sublist in self.world for x in sublist])
        im = im.copy().convert("P").resize(
            (self.width * scale, self.height * scale),
            Image.ANTIALIAS
        )
        return im
