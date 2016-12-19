from PIL import Image

from automaton.CellState import CellState
from util import gifmaker


class SimpleProcessor:
    world = None
    processing_function_bundle = None
    learning_input = False
    learning_input_location = '../tmp/l1'

    def __init__(self, world, processing_function):
        self.world = world
        self.processing_function_bundle = processing_function

    def make_cycles(self, n):
        for x in range(0, n):
            self.make_cycle()

    def enable_learning_output(self, enable=True, learning_input_location='../tmp/l1'):
        self.learning_input = enable
        self.learning_input_location = learning_input_location

    def make_cycle(self):
        f = None
        if self.learning_input:
            f = open(self.learning_input_location, 'a')
        self.world.prepare_copy()
        for i in range(0, self.world.height):
            for j in range(0, self.world.width):
                if self.learning_input:
                    output = self.processing_function_bundle[0](self.world, i, j, self.learning_input)
                    state = output[- 1]
                    self.world.set_in_world_copy(i, j, state)
                    if state == CellState.Alive:
                        output[- 1] = 1
                    else:
                        output[- 1] = 0
                    if f:
                        f.write(' '.join([str(item) for item in output]) + '\n')
                else:
                    self.world.set_in_world_copy(i, j, self.processing_function_bundle[0](self.world, i, j))
        self.world.switch_copy()

    def make_cycles_gif(self, n, path, scale=1):
        sequence = []
        for x in range(0, n + 1):
            if x != 0:
                self.make_cycle()
            im = self.world.get_as_image()
            sequence.append(
                im.copy().convert("P").resize(
                    (self.world.width * scale, self.world.height * scale),
                    Image.ANTIALIAS
                )
            )
        gif_file = open(path, "wb")
        gifmaker.make_delta(gif_file, sequence)
        gif_file.close()
