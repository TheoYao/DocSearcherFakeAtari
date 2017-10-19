# encoding: utf-8

import numpy as np
import pandas as pd
# import time

np.random.seed(2)  # reproducible


N_STATES = 20
ACTIONS = ['choose', 'skip']
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 20
FRESH_TIME = 0.3


def make_observation(cur_pos, episode, step_counter):
    env_list = ['*'] * cur_pos + ['-'] * (N_STATES - cur_pos) + ['T']
    tips = 'Episode %s: total_steps: %s' % (episode+1, step_counter)
    # print(tips)
    # print(''.join(env_list))
    # time.sleep(1)


def make_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name


def make_policy(cur_pos, action, is_pos, q_table):
    if action == 'choose':
        # print('chosen!')
        # return (
        #     cur_pos+1,
        #     1 + q_table.ix[cur_pos, 'choose']
        # ) if is_pos else (cur_pos+1, -1 + q_table.ix[cur_pos, 'choose'])
        return (cur_pos+1, 1) if is_pos else (cur_pos+1, -1)
    else:
        # print('skip!')
        # return (cur_pos, q_table.ix[cur_pos, 'skip'])
        return (cur_pos, 0)


def reset_env():
    return 0, 0


def eposide_refresh_q_table(q_table, total_rewared):
    pass


def gen_q_table(pos_docs, neg_docs):
    q_table = pd.DataFrame(
        np.zeros((N_STATES+1, len(ACTIONS))),
        columns=ACTIONS,
    )
    chosen_docs = []
    step_counter = 0
    cur_pos = 0
    pos_index = 0
    neg_index = 0
    # for turn in range(10):
    print('EPSION', '\t', 'step_counter', '\t', 'total_reward')
    for episode in range(MAX_EPISODES):
        is_terminated = False
        total_reward = 0
        # global EPSILON
        # if EPSILON < 0.8:
        #     EPSILON += 0.1
        while not is_terminated:
            make_observation(cur_pos, episode, step_counter)
            # while not is_terminated:
            # if np.random.randint(0, 2):
            if step_counter % 2:
                cur_doc = pos_docs[pos_index]
                pos_index = pos_index + 1
                is_pos = 1
            else:
                cur_doc = neg_docs[neg_index]
                neg_index = neg_index + 1
                is_pos = 0
            action = make_action(cur_pos, q_table)
            next_pos, reward = make_policy(cur_pos, action, is_pos, q_table)
            total_reward += reward
            q_predict = q_table.ix[next_pos, action]
            if next_pos < N_STATES:
                q_target = reward + GAMMA * q_table.iloc[next_pos, :].max()
            else:
                q_target = reward
                is_terminated = True
            if is_terminated:
                break
            q_table.ix[cur_pos, action] += ALPHA * (q_target - q_predict)

            if action == 'choose':
                chosen_docs.append(cur_doc)
            cur_pos = next_pos
            # print(cur_pos)
            step_counter += 1
        print(EPSILON, '\t\t', step_counter, '\t\t',  total_reward)
        cur_pos, step_counter = reset_env()

    return q_table

if __name__ == "__main__":
    # candi_doc = 'data/candi_doc.pkl'
    candi_docs = {
        "0": [],
        "1": [],
    }
    for i in range(20000 * N_STATES):
        index = i % 2
        candi_docs[str(index)].append([i])
    pos_docs = candi_docs["1"]
    neg_docs = candi_docs["0"]
    q_table = gen_q_table(pos_docs, neg_docs)
    print('\r\nQ-table:\n')
    print(q_table)
