from automaton import RuleParser
from automaton.SimpleProcessor import SimpleProcessor
from automaton.World import World


class Main:
    @staticmethod
    def main():
        print("Starting")

        world = World(25, 25)

        world.make_random(65)

        # world.set_in_world(1, 0, CellState.Alive)
        # world.set_in_world(2, 1, CellState.Alive)
        # world.set_in_world(0, 2, CellState.Alive)
        # world.set_in_world(1, 2, CellState.Alive)
        # world.set_in_world(2, 2, CellState.Alive)
        #
        # world.set_in_world(10, 11, CellState.Alive)
        # world.set_in_world(11, 10, CellState.Alive)
        # world.set_in_world(12, 10, CellState.Alive)
        # world.set_in_world(12, 11, CellState.Alive)
        # world.set_in_world(12, 12, CellState.Alive)

        processing_function = RuleParser.parse_rule_file('../resource/rule/2DA/life')

        processor = SimpleProcessor(world, processing_function)

        processor.enable_learning_input()

        world.print()

        processor.make_cycles_gif(1, '../tmp/w0.gif', 5)

        world.save_as_image('../tmp/w0.png')

        print("")
        #
        # for i in range(1, 20):
        #     processor.make_cycles(1)
        #     world.save_as_image('tmp/w' + str(i) + '.png')

        world.print()

        # world.save_as_image("tmp/w1.png")


if __name__ == '__main__':
    Main.main()
