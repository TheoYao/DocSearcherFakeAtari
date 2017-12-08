# encoding: utf-8
import csv


class StatMod:
    def __init__(self, average_stage):
        self.average_stage = average_stage

        self.average_q_values = []
        self.rewards = []
        self.is_optimal = []

    def output(self):
        with open('average_q_values.csv', 'wb') as csv_file:
            csv_writer = csv.writer(csv_file)
            for i in range(len(self.average_q_values / self.average_stage)):
                csv_writer.writerow(
                    self.average_q_values[i*self.average_stage,
                                          (i+1)*self.average_stage]
                )
