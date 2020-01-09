# Author: DingQinbin
# -*- coding: utf-8 -*-
# Time: 2018/10/6

import pandas as pd
columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex",
           "capital_gain", "capital_loss", "hours_per_week", "native_country", "high_income"]
income = pd.read_csv("income.csv", names = columns)
print(income.head(5))

print(income["workclass"].unique())
col = pd.Categorical(income["workclass"])
print(col)
income["workclass"] = col.codes
# print(income["workclass"].head(5))
for name in ["education", "marital_status", "occupation", "relationship", "race", "sex", "native_country", "high_income"]:
    col = pd.Categorical(income[name])
    income[name] = col.codes

import math
entropy = -(2/5.0 * math.log(2/5.0) + 3/5 * math.log(3/5.0, 2))
print(entropy)
prob_0 = income[income["high_income"] == 0].shape[0] / float(income.shape[0])
prob_1 = income[income["high_income"] == 1].shape[0] / float(income.shape[0])
income_entropy = -(prob_0 * math.log(float(prob_0), 2) + prob_1 * math.log(float(prob_1), 2))

import numpy as np
def calc_entropy(column):
    counts = np.bincount(column)
    probabilities = counts / float(len(column))

    entropy = 0
    for prob in probabilities:
        if prob > 0:
            entropy += prob * math.log(float(prob), 2)

    return -entropy
# entropy = calc_entropy([1, 1, 0, 0, 1])
# print(entropy)
information_gain = entropy - ((8 * calc_entropy([1, 1, 0, 0])) + (.2 * calc_entropy([1])))
print(information_gain)
print(income["high_income"])

income_entropy = calc_entropy(income["high_income"])
print(income_entropy)
median_age = income["age"].median()

left_split = income[income["age"] <= median_age]
right_split = income[income["age"] > median_age]

age_information_gain = income_entropy - ((left_split.shape[0] / income.shape[0]) * calc_entropy(left_split["high_income"]) +
                                         ((right_split.shape[0] / income.shape[0]) * calc_entropy(right_split["high_income"])))
# 信息增益
# print(age_information_gain)


def calc_information_gain(data, split_name, target_name):
    original_entropy = calc_entropy(data[target_name])
    column = data[split_name]
    median = column.median()

    left_split = data[column <= median]
    right_split = data[column > median]

    to_subtract = 0
    for subset in [left_split, right_split]:
        prob = (subset.shape[0] / float(data.shape[0]))
        to_subtract += prob * calc_entropy(subset[target_name])

    return original_entropy - to_subtract


print(calc_information_gain(income, "age", "high_income"))

columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship",
          "race", "sex",
          "capital_gain", "capital_loss", "hours_per_week", "native_country", "high_income"]
information_gains = []
for col in columns:
    information_gain = calc_information_gain(income, col, "high_income")
    information_gains.append(information_gain)

highest_gain_index = information_gains.index(max(information_gains))
highest_gain = columns[highest_gain_index]
print(highest_gain)

# 选出信息增益最大的那个特征
def find_best_column(data, terget_name, columns):
    information_gains = []
    for col in columns:
        information_gain = calc_information_gain(data, col, "high_income")
        information_gains.append(information_gain)

    highest_gain_index = information_gains.index(max(information_gains))
    return highest_gain

income_split = find_best_column(income, "high_income", columns)
print(income_split)

# 选取中位数进行切分
def id3(data, target, columns):
    unique_targets = pd.unique(data[target])

    if len(unique_targets) == 1:
        if 0 in unique_targets:
            label_0s.append(0)
        elif 1 in unique_targets:
            label_1s.append(1)
        return
    best_column = find_best_column(data, target, columns)
    column_median = data[best_column].median()

    left_split = data[data[best_column] <= column_median]
    right_split = data[data[best_column] > column_median]

    for split in [left_split, right_split]:
        id3(split, target, columns)
income[['marital_status', 'age', 'high_income']] = income[['marital_status', 'age', 'high_income']].astype('int32')
id3(income, "high_income", ["age", "marital_status"])

tree = {}
nodes = []
def id3(data, target, columns, tree):
    unique_targets = pd.unique(data[target])
    nodes.append(len(nodes) + 1)
    tree["number"] = nodes[-1]

    if len(unique_targets) == 1:
        if 0 in unique_targets:
            tree["label"] = 0
        elif 1 in unique_targets:
            tree["label"] = 1
        return

    best_column = find_best_column(data, target, columns)
    column_median = data[best_column].median()

    tree["column"] = best_column
    tree["median"] = column_median

    left_split = data[data[best_column] <= column_median]
    right_split = data[data[best_column] > column_median]
    split_dict = [["left", left_split], ["right", right_split]]

    for name, split in split_dict:
        tree[name] = {}
        id3(split, target, columns, tree[name])

id3(income, "high_income", ["age", "marital_status"], tree)

def print_with_depth(string, depth):
    prefix = "   " * depth
    print("{0}{1}".format(prefix, string))

def print_node(tree, depth):
    if "label" in tree:
        print_with_depth("Leaf: Label {0}".format(tree["label"]), depth)
        return
    print_with_depth("{0} > {1}".format(tree["column"], tree["median"]), depth)
    branches = [tree["left"], tree["right"]]

print_node(tree, 0)

# 保存树的信息
def print_node(tree, depth):
    if "label" in tree:
        print_with_depth("Leaf: Label {0}".format(tree["label"]), depth)
        return
    print_with_depth("{0} > {1}".format(tree["column"], tree["median"]), depth)
    for branch in [tree["left"], tree["right"]]:
        print_node(branch, depth + 1)

print_node(tree, 0)

def predict(tree, row):
    if "label" in tree:
        return tree["label"]

    column = tree["column"]
    median = tree["median"]

print(predict(tree, data.iloc[0]))
def predict(tree, row):
    if "label" in tree:
        return tree["label"]

    column = tree["column"]
    median = tree["median"]
    if row[column] <= median:
        return predict(tree["left"], row)
    else:
        return predict(tree["right"], row)

print(predict(tree, data.iloc[0]))

new_data = pd.DataFrame([40,0], [20,2], [80,1], [15,1], [27,2], [38,1])
new_data.columns = ["age", "marital_status"]

def batch_predict(tree, df):
    pass

predictions = batch_predict(tree, new_data)
def batch_predict(tree, df):
    return df.apply(lambda x: predict(tree, x), axis = 1)

predictions = batch_predict(tree, new_data)
print(predictions)