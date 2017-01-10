from automaton import RuleParser
from automaton.SimpleProcessor import SimpleProcessor
from automaton.World import World


def load_training_set(file_path):
    with open(file_path) as f:
        data = [[float(i) for i in l.split()] for l in f.readlines() if
                (not l.startswith('#')) and (not l.strip() == '')]
        data = [[row[:-1] for row in data], [row[-1] for row in data]]
        return data


def reduce_training_set(training_set):
    n = 0
    training_set_tmp_inputs = []
    training_set_tmp_outputs = []
    for row in training_set[0]:
        if row not in training_set_tmp_inputs:
            training_set_tmp_inputs.append(row)
            training_set_tmp_outputs.append(training_set[1][n])
        n += 1
    return [training_set_tmp_inputs, training_set_tmp_outputs]


def save_training_set(training_set, file_out):
    with open(file_out, 'w') as f:
        for i in range(len(training_set[0])):
            f.write(' '.join([str(item) for item in (training_set[0][i] + [training_set[1][i]])]) + '\n')


def generate_reduced_training_set_from_cellular(processing_function_rule_location, world_size=None, world_percentage=65,
                                                file_learn_loc_tmp='tmp_file_tmp', cycles_count=100):
    if world_size is None:
        world_size = [100, 100]

    world = World(world_size[0], world_size[1])
    processing_function = RuleParser.parse_rule_file(processing_function_rule_location)
    world.make_random(world_percentage)
    processor = SimpleProcessor(world, processing_function)
    processor.enable_learning_output(True, file_learn_loc_tmp)
    processor.clear_learning_output()
    processor.make_cycles(cycles_count)
    training_set = load_training_set(file_learn_loc_tmp)
    training_set = reduce_training_set(training_set)
    return training_set
