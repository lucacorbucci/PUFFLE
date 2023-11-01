from collections import Counter

import dill

nodes_with_disparity = dill.load(open("./nodes_with_disparity.pkl", "rb"))

print(len(nodes_with_disparity))


# read all the lines from the file
counters = []
with open("./outpuit.txt") as fp:
    for line in fp:
        # split the line into words
        words = line.split(" -  ")
        # remove \n from the last word
        words[1] = words[1].replace("\n", "")
        # convert the string to a counter
        counter = eval(words[1])
        # append the counter to the list
        counters.append(counter)


def compute_disparity(counter):
    Counter({(0, 1): 337, (1, 0): 233, (0, 0): 222, (1, 1): 110})

    count_sensitive_value_1 = 0
    count_sensitive_value_0 = 0
    for key, value in counter.items():
        if key[1] == 1:
            count_sensitive_value_1 += value
        else:
            count_sensitive_value_0 += value

    sensitives = {1: count_sensitive_value_1, 0: count_sensitive_value_0}

    disparities = []
    for target in [0, 1]:
        for sensitive_value in [0, 1]:
            count_target_sensitive = counter[(target, sensitive_value)]
            count_target_sensitive_opposite = counter[
                (target, 1 if sensitive_value == 0 else 0)
            ]

            disparity = (count_target_sensitive) / (sensitives[sensitive_value]) - (
                count_target_sensitive_opposite
            ) / (sensitives[1 if sensitive_value == 0 else 0])

            disparities.append(disparity)
    return max(disparities)


disparities = [compute_disparity(counter) for counter in counters]


selected = {}
not_selected = {}
for node_name, disparity in nodes_with_disparity.items():
    for disparity_value, counter in zip(disparities, counters):
        if round(disparity_value, 5) == round(disparity, 5):
            print(node_name, disparity, counter)
            selected[node_name] = (disparity, counter)
        else:
            not_selected[node_name] = (disparity, counter)

print(len(selected))
print(len(not_selected))
dill.dump(selected, open("./selected.pkl", "wb"))
dill.dump(not_selected, open("./not_selected.pkl", "wb"))
