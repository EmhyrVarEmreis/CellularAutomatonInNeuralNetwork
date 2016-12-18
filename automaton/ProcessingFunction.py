from automaton import Neighborhood
from automaton.CellState import CellState


def get_processing_function(dimensions, neighborhood_type, birth_nums, survival_nums):
    neighborhood = None
    processing_type = dimensions + '' + neighborhood_type
    if processing_type == '2A':
        neighborhood = Neighborhood.moore2d
    elif processing_type == '2B':
        neighborhood = Neighborhood.neumann2d

    def processing_function(world, x, y, extended_output=False):
        output = neighborhood(world, x, y, extended_output)
        n = output[len(output) - 1]
        if world.get_cell(x, y) == CellState.Alive:
            if n in survival_nums:
                new_state = CellState.Alive
            else:
                new_state = CellState.Dead
        else:
            if n in birth_nums:
                new_state = CellState.Alive
            else:
                new_state = CellState.Dead
        if extended_output:
            output[len(output) - 1] = new_state
            return output
        else:
            return new_state

    return processing_function