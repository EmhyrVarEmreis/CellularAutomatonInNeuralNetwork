from CellState import CellState


def moore2d(world, x, y):
    n = 0
    for k in range(x - 1, x + 2):
        for l in range(y - 1, y + 2):
            if world.get_cell(k, l) == CellState.Alive:
                n += 1
    if world.get_cell(x, y) == CellState.Alive:
        n -= 1
    return n


def neumann2d(world, x, y):
    n = 0
    if world.get_cell(x - 1, y) == CellState.Alive:
        n += 1
    if world.get_cell(x + 1, y) == CellState.Alive:
        n += 1
    if world.get_cell(x, y - 1) == CellState.Alive:
        n += 1
    if world.get_cell(x, y + 1) == CellState.Alive:
        n += 1
    return n


def get_neighborhood_function(neighborhood_type):
    if neighborhood_type == 'A2':
        return moore2d
    if neighborhood_type == 'B2':
        return neumann2d
