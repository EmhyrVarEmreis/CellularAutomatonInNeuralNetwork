from automaton.CellState import CellState


def moore2d(world, x, y, extended_output):
    n = 0
    output = []
    for k in range(x - 1, x + 2):
        for l in range(y - 1, y + 2):
            state = world.get_cell(k, l)
            output.append(state)
            if state == CellState.Alive:
                n += 1
    if world.get_cell(x, y) == CellState.Alive:
        n -= 1
    if extended_output:
        output.append(n)
        return output
    return [n]


def neumann2d(world, x, y, extended_output):
    n = 0
    output = [
        world.get_cell(x - 1, y),
        world.get_cell(x, y - 1),
        world.get_cell(x, y),
        world.get_cell(x, y + 1),
        world.get_cell(x + 1, y)
    ]
    for state in output:
        if state == CellState.Alive:
            n += 1
    if output[2] == CellState.Alive:
        n -= 1
    if extended_output:
        output.append(n)
        return output
    return [n]


def get_neighborhood_function(neighborhood_type):
    if neighborhood_type == 'A2':
        return moore2d
    if neighborhood_type == 'B2':
        return neumann2d
