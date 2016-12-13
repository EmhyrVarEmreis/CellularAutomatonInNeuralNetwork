from CellState import CellState

from automaton import Neighborhood


def moore2d(world, x, y):
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


def neumann2d(world, x, y):
    n = 0
    for k in range(x - 1, x + 2):
        for l in range(y - 1, y + 2):
            if world.get_cell(k, l) == CellState.Alive:
                n += 1
    if world.get_cell(x, y) == CellState.Alive:
        n -= 1
    return n


def get_processing_function(dimensions, neighborhood_type, birth_nums, survival_nums):
    neighborhood = None
    processing_type = dimensions + '' + neighborhood_type
    if processing_type == '2A':
        neighborhood = Neighborhood.moore2d
    elif processing_type == '2B':
        neighborhood = Neighborhood.neumann2d

    def processing_function(world, x, y):
        n = neighborhood(world, x, y)
        if world.get_cell(x, y) == CellState.Alive:
            if n in survival_nums:
                return CellState.Alive
            else:
                return CellState.Dead
        else:
            if n in birth_nums:
                return CellState.Alive
            else:
                return CellState.Dead

    return processing_function
