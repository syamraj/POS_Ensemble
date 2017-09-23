import random
with open('/home/devil/Thesis/opennlp-mixed-train-data', 'r') as source:
    data = [ (random.random(), line) for line in source]
data.sort()
with open('/home/devil/Thesis/opennlp-mixed_and_shuffled-train-data', 'w') as target:
    for _, line in data:
        target.write(line)