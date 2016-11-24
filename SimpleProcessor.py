from CellState import CellState


class SimpleProcessor:
    world = None

    def __init__(self, world):
        self.world = world

    def make_cycle(self):
        self.world.prepare_copy()
        for i in range(0, self.world.height):
            for j in range(0, self.world.width):
                n = 0
                for k in range(i - 1, i + 2):
                    for l in range(j - 1, j + 2):
                        if self.world.get_cell(k, l) == CellState.Alive:
                            n += 1
                state = self.world.get_cell(i, j)
                if (state == CellState.Alive and (n == 4 or n == 3)) or (state == CellState.Dead and n == 3):
                    self.world.set_in_world_copy(i, j, CellState.Alive)
                else:
                    self.world.set_in_world_copy(i, j, CellState.Dead)
        self.world.switch_copy()
