# encoding: utf-8

import numpy as np
from neural_dqn import NeuralDQN
from fake_atari import DocSeq

ACTIONS = ["choose", "skip"]
TO_SORT_NUMS = 20


def lets_begin():
    global ACTIONS
    actions = len(ACTIONS)
    agent = NeuralDQN(actions)
    sorter = DocSeq(TO_SORT_NUMS)
    agent.set_init_state(np.zeros((TO_SORT_NUMS, 100, 60)))
    doc_cursor = 0
    while 1:
        action = agent.make_action(doc_cursor)
        # if action[0] > 0:
        #     action = ACTIONS.index('choose')
        # else:
        #     action = ACTIONS.index('skip')
        observation, reward, is_over, doc_cursor = sorter.next(action)
        agent.set_perception(observation, action, reward, is_over, doc_cursor)


if __name__ == "__main__":
    lets_begin()
