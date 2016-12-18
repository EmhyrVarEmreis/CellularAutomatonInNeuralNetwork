from numpy import array

from automaton import RuleParser
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
    file_learn_loc = '../tmp/l1'
    file_output_loc = '../tmp/n1'
    learn_cycles_count = 10000
    world_size = [25, 25]
    world_percentage = 65
    cycles_count = 1
    world_gif_location = '../tmp/w1.gif'
    processing_function_rule_location = '../resource/rule/2DA/life'

    world = World(world_size[0], world_size[1])
    world.make_random(world_percentage)
    processing_function = RuleParser.parse_rule_file(processing_function_rule_location)
    processor = SimpleProcessor(world, processing_function)
    processor.enable_learning_input(True, file_learn_loc)

    if world_gif_location is not None:
        processor.make_cycles_gif(cycles_count, world_gif_location, 5)
    else:
        processor.make_cycles(cycles_count)

    neural_network_combined = learn_from_file(file_learn_loc, file_output_loc, learn_cycles_count)

    print("Status: " + str(neural_network_combined[1]) + '%')
