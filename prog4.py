import math

import numpy as np
import pandas as pd

data = pd.read_csv('play.csv')
features = [feat for feat in data]
features.remove('classification')


class Node:
    def __init__(self):
        self.children = []
        self.value = ''
        self.isLeaf = False
        self.pred = ''


def entropy(example):
    pos = 0.0
    neg = 0.0
    for _, row in example.iterrows():
        if row['classification'] == 'Yes':
            pos += 1
        else:
            neg += 1
    if pos == 0.0 or neg == 0:
        return 0
    p = pos / (pos + neg)
    n = neg / (pos + neg)
    entropy = -(p * math.log(p, 2) + n * math.log(n, 2))
    return entropy


def info_gain(example, attr):
    uniq = np.unique(example[attr])
    gain = entropy(example)
    for u in uniq:
        subdata = example[example[attr] == u]
        sub_e = entropy(subdata)
        gain -= (float(len(subdata)) / float(len(example))) * sub_e
    return gain


def ID3(example, attr):
    root = Node()
    max_gain = 0
    max_feat = ''
    for feature in attr:
        gain = info_gain(example, feature)
        if gain > max_gain:
            max_gain = gain
            max_feat = feature
    root.value = max_feat
    uniq = np.unique(example[max_feat])
    for u in uniq:
        subdata = example[example[max_feat] == u]
        if entropy(subdata) == 0.0:
            newNode = Node()
            newNode.value = u
            newNode.isLeaf = True
            newNode.pred = np.unique(subdata['classification'])
            root.children.append(newNode)
        else:
            dummyNode = Node()
            dummyNode.value = u
            new_attr = attr.copy()
            new_attr.remove(max_feat)
            child = ID3(subdata, new_attr)
            dummyNode.children.append(child)
            root.children.append(dummyNode)
    return root


def print_tree(root, depth=0):
    for i in range(depth):
        print('\t', end="")
    print(root.value, end='')
    if root.isLeaf:
        print('->',root.pred)
    print()
    for child in root.children:
        print_tree(child, depth + 1)


root = ID3(data, features)
print_tree(root)
