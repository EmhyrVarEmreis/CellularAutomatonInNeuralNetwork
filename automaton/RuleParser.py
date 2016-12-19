from automaton import ProcessingFunction


def parse_rule_file(file_name):
    with open(file_name) as f:
        content = f.readlines()
        content = [line.strip() for line in content if (not line.startswith('#')) and (not line.strip() == '')]
        return ProcessingFunction.get_processing_function_bundle(
            content[0], content[1], list(map(int, content[2].split())), list(map(int, content[3].split()))
        )
