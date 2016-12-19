from numpy import array

from automaton import RuleParser
from automaton.ProcessingFunction import get_neural_processing_function_bundle
from automaton.SimpleProcessor import SimpleProcessor
from automaton.World import World
from neural import TrainingLoader
from neural.SimpleNeuralNetwork import SimpleNeuralNetwork


def learn_from_file(file_learn, file_output=None, cycles=10000):
    training_set_tmp = TrainingLoader.load(file_learn)
    network = SimpleNeuralNetwork(len(training_set_tmp[0][0]))
    network.train(array(training_set_tmp[0]), array([training_set_tmp[1]]).T, cycles)
    if file_output is not None:
        network.save_synaptic_weights(file_output)
    status = network.verify(training_set_tmp)
    return [network, status, training_set_tmp]


if __name__ == "__main__":
    # Options
    file_learn_loc = '../tmp/l1'
    file_output_loc = '../tmp/n1'
    learn_cycles_count = 10
    world_size = [25, 25]
    world_percentage = 65
    world_location = '../tmp/w1.txt'
    cycles_count_learning = 1
    cycles_count_normal = 100
    cycles_count_neural = 100
    gif_location_normal = '../tmp/w1a.gif'
    gif_location_neural = '../tmp/w1b.gif'
    processing_function_rule_location = '../resource/rule/2DA/life'

    # Init world
    world = World(25, 25)
    world.make_random(world_percentage)

    # Load processing function
    processing_function = RuleParser.parse_rule_file(processing_function_rule_location)
    processor = SimpleProcessor(world, processing_function)

    # Enable learning output
    processor.enable_learning_output(True, file_learn_loc)

    # Run learning cycles
    processor.make_cycles(cycles_count_learning)

    # Learn network
    neural_network_combined = learn_from_file(file_learn_loc, file_output_loc, learn_cycles_count)

    # Print success percentage
    print("Status: " + str(neural_network_combined[1]) + '%')

    # Disable learning output
    processor.enable_learning_output(False)

    # Init new world
    world.clear()
    world.make_random(world_percentage)

    # Save world before processing
    world.save(world_location)

    # Make normal cycles GIF
    processor.make_cycles_gif(cycles_count_normal, gif_location_normal, 5)

    # TODO set neural_processing_function
    processing_function_neural = get_neural_processing_function_bundle(processor.processing_function_bundle[1],
                                                                       neural_network_combined[0])
    processor = SimpleProcessor(world, processing_function_neural)

    # Load the same world
    world.load(world_location)
    world.print()

    # Make neural processed cycles GIF
    processor.make_cycles_gif(cycles_count_normal, gif_location_neural, 5)
