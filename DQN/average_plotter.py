# encoding: utf-8

import matplotlib.pyplot as plt
import csv
# import sys
from scipy import optimize
import numpy as np


def linear(x, A, B):
    return A * x + B


def exp_nature(x, A, B):
    return A * np.exp(B/x)


def average_cal(name, stage=10):
    with open(name+'.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        result = []
        for row in csv_reader:
            cur_row_sum = 0
            for item in row:
                if not item:
                    continue
                cur_row_sum += int(item)
            result.append(cur_row_sum/stage)
            if len(result) > 200:
                break
    plt.scatter(range(1, len(result)+1), result)
    plt.show()


def draw_this():
    stage = 10
    averages = [0, 0]
    choose = []
    skip = []
    with open('average_q_values.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            for i in range(0, stage):
                trans_list = row[i][1:-1].split(' ')
                useful_list = []
                for item in trans_list:
                    if item is not '':
                        useful_list.append(item)
                averages[0] += float(useful_list[0])
                averages[1] += float(useful_list[1])
            choose.append(averages[0] / stage)
            skip.append(averages[1] / stage)
            averages = [0, 0]
    plt.scatter(range(0, len(choose)), skip, c='blue', marker='x')

    # A1, B1 = optimize.curve_fit(linear, range(0, len(choose)), skip)[0]
    A1, B1 = optimize.curve_fit(exp_nature, range(1, len(choose)+1), skip)[0]
    x1 = np.arange(1, len(skip)+1)
    # y1 = linear(x1, A1, B1)
    y1 = exp_nature(x1, A1, B1)
    plt.plot(x1, y1)
    plt.show()


def draw_that():
    average_cal('average_rewards')
    average_cal('is_optimal')

if __name__ == '__main__':
    draw_this()
    draw_that()
