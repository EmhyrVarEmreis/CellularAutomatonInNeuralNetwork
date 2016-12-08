from PIL import Image

from util import gifmaker


class SimpleProcessor:
    world = None
    processing_function = None

    def __init__(self, world, processing_function):
        self.world = world
        self.processing_function = processing_function

    def make_cycles(self, n):
        for x in range(0, n):
            self.make_cycle()

    def make_cycle(self):
        self.world.prepare_copy()
        for i in range(0, self.world.height):
            for j in range(0, self.world.width):
                self.world.set_in_world_copy(i, j, self.processing_function(self.world, i, j))
        self.world.switch_copy()

    def make_cycles_gif(self, n, path, scale=1):
        sequence = []
        for x in range(0, n):
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
