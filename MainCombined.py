import random

from numpy import array

from automaton import RuleParser
from automaton.CellState import CellState
from automaton.ProcessingFunction import get_neural_processing_function_bundle
from automaton.SimpleProcessor import SimpleProcessor
from automaton.World import World
from neural import TrainingLoader
from neural.SimpleLayeredNeuralNetwork import SimpleLayeredNeuralNetwork
from neural.SimpleNeuralNetwork import SimpleNeuralNetwork


def learn_from_file(file_learn, file_output=None, cycles=10000, reduce=False, multi=False, neural_num=3):
    training_set_tmp = TrainingLoader.load(file_learn)
    if reduce:
        n = 0
        c = 0
        training_set_tmp_inputs = []
        training_set_tmp_outputs = []
        for row in training_set_tmp[0]:
            if row not in training_set_tmp_inputs:
                training_set_tmp_inputs.append(row)
                training_set_tmp_outputs.append(training_set_tmp[1][n])
            n += 1
        training_set_tmp = [training_set_tmp_inputs, training_set_tmp_outputs]
    if multi:
        network = SimpleLayeredNeuralNetwork()
        network.add_layer(neural_num, len(training_set_tmp[0][0]))
        network.add_layer(1, neural_num)
        layers = network.layers
    else:
        network = SimpleNeuralNetwork(len(training_set_tmp[0][0]))
        layers = network.synaptic_weights
    print(len(training_set_tmp[1]))
    network.print_weights()
    network.train(array(training_set_tmp[0]), array([training_set_tmp[1]]).T, cycles)
    network.print_weights()
    # for cycle in range(cycles):
    #     network.train(array(training_set_tmp[0]), array([training_set_tmp[1]]).T, 1)
    #     print(network.verify(training_set_tmp))
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
    world_size = [25, 25]
    world_percentage = 55
    world_location = 'tmp/w1.txt'
    world_location_old = 'tmp/w0.txt'
    cycles_count_learning = 250
    cycles_count_normal = 150
    cycles_count_neural = 150
    gif_location_normal = 'tmp/w1a.gif'
    gif_location_neural = 'tmp/w1b.gif'
    processing_function_rule_location = 'resource/rule/2DA/life'
    neural_multi = True
    neural_multi_layer1_count = 9

    random.seed(1)

    world = World(world_size[0], world_size[1])
    processing_function = RuleParser.parse_rule_file(processing_function_rule_location)

    # Init world
    world.make_random(world_percentage)
    world.set_in_world(3, 3, CellState.Alive)
    world.set_in_world(3, 4, CellState.Alive)
    world.set_in_world(3, 5, CellState.Alive)
    world.save(world_location_old)
    world.save_as_image('tmp/0.png', 5)

    # Load processing function
    processor = SimpleProcessor(world, processing_function)

    # # Clear learning output
    # processor.clear_learning_output()

    # # Enable learning output
    # processor.enable_learning_output(True, file_learn_loc)

    # # Run learning cycles
    # processor.make_cycles(cycles_count_learning)

    # Load network
    nn = SimpleLayeredNeuralNetwork()
    nn.read_synaptic_weights('resource/learned/0.0159669230214.txt')
    print(nn.verify(TrainingLoader.load('tmp/life_all')))
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
    processor.make_cycles_gif(cycles_count_normal, gif_location_normal, 5)
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
    processor.make_cycles_gif(cycles_count_neural, gif_location_neural, 5)
    # world.save_as_image('tmp/b1.png', 5)
    # processor.make_cycle()
    # world.save_as_image('tmp/b2.png', 5)
    # processor.make_cycle()
    # world.save_as_image('tmp/b3.png', 5)
    # processor.make_cycle()
    # world.save_as_image('tmp/b4.png', 5)
    # processor.make_cycle()
    # world.save_as_image('tmp/b5.png', 5)
