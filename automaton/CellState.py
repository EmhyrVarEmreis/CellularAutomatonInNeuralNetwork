class CellState:
    Dead, Alive = range(2)


def cell_state_from(n):
    if n == 1:
        return CellState.Alive
    else:
        return CellState.Dead
