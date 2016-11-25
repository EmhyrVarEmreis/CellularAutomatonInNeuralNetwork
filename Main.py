from CellState import CellState
from SimpleProcessor import SimpleProcessor
from World import World


class Main:
    @staticmethod
    def main():
        print("Starting")

        world = World(25, 25)

        world.set_in_world(1, 0, CellState.Alive)
        world.set_in_world(2, 1, CellState.Alive)
        world.set_in_world(0, 2, CellState.Alive)
        world.set_in_world(1, 2, CellState.Alive)
        world.set_in_world(2, 2, CellState.Alive)

        world.set_in_world(10, 11, CellState.Alive)
        world.set_in_world(11, 10, CellState.Alive)
        world.set_in_world(12, 10, CellState.Alive)
        world.set_in_world(12, 11, CellState.Alive)
        world.set_in_world(12, 12, CellState.Alive)

        processor = SimpleProcessor(world)

        world.print()

        processor.make_cycles_gif(500, 'tmp/w0.gif', 5)

        world.save_as_image('tmp/w0.png')

        print("")
        #
        # for i in range(1, 20):
        #     processor.make_cycles(1)
        #     world.save_as_image('tmp/w' + str(i) + '.png')

        world.print()

        # world.save_as_image("tmp/w1.png")


if __name__ == '__main__':
    Main.main()
