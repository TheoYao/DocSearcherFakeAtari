# encoding: utf-8

from neural_dqn import NeuralDQN
from fake_atari import DocSeq

ACTIONS = ["choose", "skip"]
TO_SORT_NUMS = 10


def lets_begin():
    global ACTIONS
    actions = len(ACTIONS)
    agent = NeuralDQN(actions, TO_SORT_NUMS)
    sorter = DocSeq(TO_SORT_NUMS)
    agent.set_init_state(TO_SORT_NUMS)
    while 1:
        action = agent.make_action()
        # if action[0] > 0:
        #     action = ACTIONS.index('choose')
        # else:
        #     action = ACTIONS.index('skip')
        observation, reward = sorter.next(action)
        agent.set_perception(observation, action, reward)


if __name__ == "__main__":
    lets_begin()
