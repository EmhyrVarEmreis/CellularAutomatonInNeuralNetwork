import sys

from util.training import generate_reduced_training_set_from_cellular, save_training_set


def main(argv):
    opt_processing_function_rule_location = 'resource/rule/2DA/2x2'
    opt_world_size = [50, 50]
    opt_world_percentage = 65
    opt_cycles_count = 50
    opt_output_location = 'resource/training_set/2x2_all'
    if len(argv) == 5:
        opt_processing_function_rule_location = argv[0]
        opt_world_size = [int(x) for x in argv[1].split('-')]
        opt_world_percentage = int(argv[2])
        opt_cycles_count = int(argv[3])
        opt_output_location = argv[4]

    training_set = generate_reduced_training_set_from_cellular(
        opt_processing_function_rule_location,
        world_size=opt_world_size,
        world_percentage=opt_world_percentage,
        cycles_count=opt_cycles_count
    )

    size_cur = len(training_set[0])
    size_max = 2 ** len(training_set[0][0])
    percent = 100.0 * size_cur / size_max

    print('Training set size: %d of %d (%.2f%%)' % (size_cur, size_max, percent))

    save_training_set(training_set, opt_output_location)


if __name__ == "__main__":
    main(sys.argv[1:])
