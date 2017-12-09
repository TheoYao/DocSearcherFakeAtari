# encoding: utf-8
import csv


class StatMod:
    def __init__(self, average_stage):
        self.average_stage = average_stage

        self.average_q_values = []
        self.average_rewards = []
        self.is_optimal = []

    def output(self):
        # with open('average_q_values.csv', 'wb') as csv_file:
        #     csv_writer = csv.writer(csv_file)
        #     for i in range(len(self.average_q_values / self.average_stage)):
        #         csv_writer.writerow(
        #             self.average_q_values[i*self.average_stage:
        #                                   (i+1)*self.average_stage]
        #         )
        self.average_csv_output(self.average_q_values, 'average_q_values')
        self.average_csv_output(self.average_rewards, 'average_rewards')
        self.average_csv_output(self.is_optimal, 'is_optimal')

    def average_csv_output(self, output_list, output_str, stage=None):
        stage = stage or self.average_stage
        with open(output_str + '.csv', 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            for i in range(int(len(output_list) / self.average_stage)):
                csv_writer.writerow(
                    output_list[i*self.average_stage:
                                (i+1)*self.average_stage]
                )
