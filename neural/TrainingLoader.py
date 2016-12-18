def load(file_path):
    with open(file_path) as f:
        data = [[float(i) for i in l.split()] for l in f.readlines() if
                (not l.startswith('#')) and (not l.strip() == '')]
        data = [[row[:-1] for row in data], [row[-1] for row in data]]
        return data
