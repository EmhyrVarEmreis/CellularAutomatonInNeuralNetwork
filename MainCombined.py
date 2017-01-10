import random

from numpy import array

from automaton import RuleParser
from automaton.CellState import CellState
from automaton.ProcessingFunction import get_neural_processing_function_bundle
from automaton.SimpleProcessor import SimpleProcessor
from automaton.World import World
from neural.SimpleLayeredNeuralNetwork import SimpleLayeredNeuralNetwork
from neural.SimpleNeuralNetwork import SimpleNeuralNetwork
from util.training import load_training_set, reduce_training_set


def learn_from_file(file_learn, file_output=None, cycles=10000, reduce=False, multi=False, neural_num=3):
    training_set_tmp = load_training_set(file_learn)
    if reduce:
        training_set_tmp = reduce_training_set(training_set_tmp)
    if multi:
        network = SimpleLayeredNeuralNetwork()
        network.add_layer(neural_num, len(training_set_tmp[0][0]))
        network.add_layer(1, neural_num)
        layers = network.layers
    else:
        network = SimpleNeuralNetwork(len(training_set_tmp[0][0]))
        layers = network.synaptic_weights
    network.print_weights()
    network.train(array(training_set_tmp[0]), array([training_set_tmp[1]]).T, cycles)
    network.print_weights()
    if file_output is not None:
        network.save_synaptic_weights(file_output)
    status = network.verify(training_set_tmp)
    return [network, status, training_set_tmp, layers]


if __name__ == "__main__":
    # Options
    file_learn_loc = 'tmp/l1'
    file_output_loc = 'tmp/n1'
    learn_cycles_count = 60000
    learn_reduce = True
    world_size = [10, 10]
    world_percentage = 55
    world_location = 'tmp/w1.txt'
    world_location_old = 'tmp/w0.txt'
    cycles_count_learning = 250
    cycles_count_normal = 20
    cycles_count_neural = 20
    gif_location_normal = 'tmp/w1a.gif'
    gif_location_neural = 'tmp/w1b.gif'
    gif_scale = 20
    processing_function_rule_location = 'resource/rule/2DA/life'
    neural_multi = True
    neural_multi_layer1_count = 9

    random.seed(1)

    world = World(world_size[0], world_size[1])
    processing_function = RuleParser.parse_rule_file(processing_function_rule_location)

    # Init world
    # world.make_random(world_percentage)
    # Blinker
    world.set_in_world(4, 4, CellState.Alive)
    world.set_in_world(4, 5, CellState.Alive)
    world.set_in_world(4, 6, CellState.Alive)
    # Glider
    world.set_in_world(1, 0, CellState.Alive)
    world.set_in_world(2, 1, CellState.Alive)
    world.set_in_world(0, 2, CellState.Alive)
    world.set_in_world(1, 2, CellState.Alive)
    world.set_in_world(2, 2, CellState.Alive)
    world.save(world_location_old)
    world.save_as_image('tmp/0.png', 5)

    # Load processing function
    processor = SimpleProcessor(world, processing_function)

    # # Enable learning output
    # processor.enable_learning_output(True, file_learn_loc)

    # # Clear learning output
    # processor.clear_learning_output()

    # # Run learning cycles
    # processor.make_cycles(cycles_count_learning)

    # Load network
    nn = SimpleLayeredNeuralNetwork()
    nn.read_synaptic_weights('resource/learned/0.0108333399636.txt')
    print(str(nn.verify(load_training_set('resource/training_set/life_all'))) + '%')
    neural_network_combined = [nn]
    # # Learn network
    # neural_network_combined = learn_from_file(file_learn_loc, file_output_loc, learn_cycles_count, learn_reduce,
    #                                           neural_multi, neural_multi_layer1_count)
    #
    # # Print success percentage
    # print("Status: " + str(neural_network_combined[1]) + '%')

    # Disable learning output
    processor.enable_learning_output(False)

    # Init new world
    world.clear()
    # world.make_random(world_percentage)
    # world.save('tmp/tmp1.txt')

    # Save world before processing
    world.load(world_location_old)

    # Make normal cycles GIF
    processor.make_cycles_gif(cycles_count_normal, gif_location_normal, gif_scale)
    # world.save_as_image('tmp/a1.png', 5)
    # processor.make_cycle()
    # world.save_as_image('tmp/a2.png', 5)
    # processor.make_cycle()
    # world.save_as_image('tmp/a3.png', 5)
    # processor.make_cycle()
    # world.save_as_image('tmp/a4.png', 5)
    # processor.make_cycle()
    # world.save_as_image('tmp/a5.png', 5)

    processing_function_neural = get_neural_processing_function_bundle(
        processor.processing_function_bundle[1], neural_network_combined[0]
    )
    processor = SimpleProcessor(world, processing_function_neural)

    # Load the same world
    world.load(world_location_old)
    world.save('tmp/tmp2.txt')

    # Make neural processed cycles GIF
    processor.make_cycles_gif(cycles_count_neural, gif_location_neural, gif_scale)
    # world.save_as_image('tmp/b1.png', 5)
    # processor.make_cycle()
    # world.save_as_image('tmp/b2.png', 5)
    # processor.make_cycle()
    # world.save_as_image('tmp/b3.png', 5)
    # processor.make_cycle()
    # world.save_as_image('tmp/b4.png', 5)
    # processor.make_cycle()
    # world.save_as_image('tmp/b5.png', 5)
