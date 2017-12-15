# encoding: utf-8
import sys
import math
from neural_dqn import NeuralDQN
from fake_atari import DocSeq
from statistic import StatMod
from collections import deque
import matplotlib.pyplot as plt


class white_gloves(object):
    ACTIONS = ["choose", "skip"]
    TO_SORT_NUMS = 40
    AVERAGE_STAGE = 10
    actions = len(ACTIONS)

    agent = NeuralDQN(actions, TO_SORT_NUMS, AVERAGE_STAGE)
    sorter = DocSeq(TO_SORT_NUMS)
    statistician = StatMod(AVERAGE_STAGE, TO_SORT_NUMS)

    agent.set_init_state(TO_SORT_NUMS)
    test_docs = deque(maxlen=TO_SORT_NUMS)

    ndcg_q_values = []

    @classmethod
    def lets_train(cls):
        while 1:
            action, q_value = cls.agent.make_action()

            if q_value[0] > q_value[1]:
                cls.ndcg_q_values.append(q_value[0])
                cls.ndcg_q_values = cls.ndcg_q_values[-1 * cls.TO_SORT_NUMS:]
                cls.statistician.NDCG.append(cls.get_NDCG(cls.ndcg_q_values))

            cls.statistician.average_q_values.append(q_value)
            # if action[0] > 0:
            #     action = ACTIONS.index('choose')
            # else:
            #     action = ACTIONS.index('skip')
            observation, reward, is_optimal = cls.sorter.next(action)
            cls.statistician.average_rewards.append(reward)
            cls.statistician.is_optimal.append(is_optimal)
            if observation is None and reward is None:
                cls.statistician.output()
                break
            cls.agent.set_perception(observation, action, reward)

    @classmethod
    def lets_test(cls):
        cls.statistician = StatMod(cls.AVERAGE_STAGE, 'test_500_')
        plt.ion()
        time_step = 1
        go_to_stat = 0
        while not go_to_stat:
            try:
                documents = list()
                while len(documents) < white_gloves.TO_SORT_NUMS:
                    ori_document = cls.sorter.get_test_document()
                    if ori_document is None:
                        go_to_stat = 1
                        break
                    ori_document = ori_document[0]
                    document = [ori_document[1]]
                    label = ori_document[0]
                    q_value = cls.agent.get_q_values(document)[0]
                    if q_value[0] > q_value[1]:
                        print(time_step)
                        documents.append((document, q_value[0]))
                        cls.statistician.average_q_values.append(q_value[0])
                        is_optimal = 1 if label else 0
                        cls.statistician.is_optimal.append(is_optimal)

                # ndcg = white_gloves.get_NDCG(
                #     list(map(lambda x: x[-1], documents)))

                # plt.scatter(time_step, ndcg)
                # plt.draw()
                # plt.pause(0.01)
                time_step += 1
            except Exception as e:
                print(str(e))
                continue
        print(cls.statistician.average_q_values)
        cls.statistician.output()

    @staticmethod
    def get_NDCG(q_values):
        trimed_q_values = list(
            map(lambda x: 2 ** min(x / 0.2, 5) - 1, q_values)
        )
        sorted_q_values = sorted(trimed_q_values, reverse=True)
        DCG = trimed_q_values[0]
        for index, q_value in enumerate(trimed_q_values[1:]):
            DCG += (q_value / math.log(index+2, 2))
        IDCG = 0.0
        for index, q_value in enumerate(sorted_q_values[1:]):
            IDCG += (q_value / math.log(index+2, 2))
        return DCG / IDCG

if __name__ == "__main__":
    assert(len(sys.argv) > 1)
    train_or_test = sys.argv[1]
    if train_or_test == 'train':
        white_gloves.lets_train()
    elif train_or_test == 'test':
        white_gloves.lets_test()
    else:
        print('ハレルヤ')
