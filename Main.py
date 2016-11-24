from CellState import CellState
from SimpleProcessor import SimpleProcessor
from World import World


class Main:
    @staticmethod
    def main():
        print("Starting")

        world = World(5, 5)

        world.set_in_world(0, 1, CellState.Alive)
        world.set_in_world(1, 0, CellState.Alive)
        world.set_in_world(2, 1, CellState.Alive)
        world.set_in_world(1, 2, CellState.Alive)

        processor = SimpleProcessor(world)

        world.print()

        print("")

        processor.make_cycle()

        world.print()


if __name__ == '__main__':
    Main.main()
