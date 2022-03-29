import numpy as np
import pandas as pd

data = pd.read_csv('data3.csv')
print(data)
concepts = np.array(data.iloc[:, 0:-1])
target = np.array(data.iloc[:, -1])
print(concepts)
print(target)


def learn(concepts, target):
    for i, val in enumerate(target):
        if val == 'yes':
            break
    specific_h = concepts[i].copy()
    print(specific_h)
    generic_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    print(generic_h)
    for i, h in enumerate(concepts):
        print(i + 1)
        print("With instance", h)
        if target[i] == "yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = "?"
                    generic_h[x][x] = "?"

            print("Specific Value", specific_h)
            print("Generic Value", generic_h)
        if target[i] == "no":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    generic_h[x][x] = specific_h[x]

            print("Specific Value", specific_h)
            print("Generic Value", generic_h)
    indices = [i for i, val in enumerate(generic_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        generic_h.remove(['?', '?', '?', '?', '?', '?'])
    return specific_h, generic_h


s_final, g_final = learn(concepts, target)
print("final s:", s_final, sep="\n")
print("final g:", g_final, sep="\n")
