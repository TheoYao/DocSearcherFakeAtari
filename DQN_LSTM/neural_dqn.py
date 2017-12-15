# endcoding: utf-8

import tensorflow as tf
import numpy as np
import random
from collections import deque
# import matplotlib.pyplot as plt

# np.set_printoptions(threshold=np.nan)

# plt.ion()

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
REPLAY_MEMORY = 1000000
GAMMA = 0.95
BATCH_SIZE = 10


class NeuralDQN:
    def __init__(self, action_nums=2, search_amount=10, average_stage=10):
        self.replay_memory = deque()
        self.search_amount = search_amount

        DOUBLE_CHOOSE_SKIP_RATE = 4
        self.OBSERVE = search_amount * DOUBLE_CHOOSE_SKIP_RATE
        self.EXPLORE = max(1000.0, 10*search_amount)
        # self.EXPLORE = 100
        self.UPDATE_TIME = search_amount
        global BATCH_SIZE
        BATCH_SIZE = search_amount

        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.action_nums = action_nums
        self.average_stage = average_stage

        (self.state_input,
         self.rnn_seq_length,
         self.q_value,
         self.w_conv1, self.b_conv1,
         self.w_conv2, self.b_conv2,
         self.w_fc1, self.b_fc1) = self.create_q_network(reuse=False)

        (self.state_input_t,
         self.rnn_seq_length_t,
         self.q_value_t,
         self.w_conv1_t, self.b_conv1_t,
         self.w_conv2_t, self.b_conv2_t,
         self.w_fc1_t, self.b_fc1_t) = self.create_q_network(reuse=True)

        self.copy_target_q_network_operation = [
            self.w_conv1_t.assign(self.w_conv1),
            self.b_conv1_t.assign(self.b_conv1),
            self.w_conv2_t.assign(self.w_conv2),
            self.b_conv2_t.assign(self.b_conv2),
            self.w_fc1_t.assign(self.w_fc1),
            self.b_fc1_t.assign(self.b_fc1),
        ]

        self.create_training_method()

        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        checkoutpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkoutpoint and checkoutpoint.model_checkpoint_path:
            self.saver.restore(self.session,
                               checkoutpoint.model_checkpoint_path)
            print("Success loaded:", checkoutpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def create_q_network(self, reuse):
        state_in = tf.placeholder("float", [None, 100, 60, 1])
        state_input = tf.reshape(state_in, [-1, 100, 60, 1])
        rnn_seq_length = tf.placeholder(tf.int32)

        w_conv1 = self.weight_variable([1, 30, 1, 32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(state_input, w_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        w_conv2 = self.weight_variable([1, 30, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, w_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x1(h_conv2)

        h_pool2_flat = tf.reshape(h_pool2, [-1, 98 * 1 * 64])
        pred = self.RNN(h_pool2_flat, reuse, rnn_seq_length)

        w_fc1 = self.weight_variable([98 * 1 * 64, 1024])
        b_fc1 = self.bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(pred, w_fc1) + b_fc1)

        w_fc2 = self.weight_variable([1024, self.action_nums])
        b_fc2 = self.bias_variable([self.action_nums])

        q_value = tf.matmul(h_fc1, w_fc2) + b_fc2

        return (state_input, rnn_seq_length, q_value, w_conv1, b_conv1,
                w_conv2, b_conv2, w_fc1, b_fc1,)

    def copy_target_q_network(self):
        self.session.run(self.copy_target_q_network_operation)

    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_nums])
        self.y_input = tf.placeholder("float", [None])
        q_action = tf.reduce_sum(tf.multiply(self.q_value, self.action_input),
                                 reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - q_action))
        self.train_step = tf.train.RMSPropOptimizer(
            0.00025, 0.99, 0.0, 1e-6).minimize(self.cost)

    def set_perception(self, next_observation, action, reward):
        new_state = next_observation if (
            len(next_observation)) > 0 else self.current_state
        self.replay_memory.append(
            (list(self.current_state), action,
             reward, list(new_state))
        )
        if len(self.replay_memory) > REPLAY_MEMORY:
            self.replay_memory.popleft()
        if self.time_step > self.OBSERVE:
            self.train_q_network()

        state = ""
        if self.time_step <= self.OBSERVE:
            state = "waiting"
        elif (self.time_step > self.OBSERVE and
              self.time_step <= self.OBSERVE + self.EXPLORE):
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", self.time_step, "/ STATE", state,
              "/ EPSILON", self.epsilon)
        self.current_state = new_state
        self.time_step += 1

    def train_q_network(self):
        mini_batch = random.sample(self.replay_memory, BATCH_SIZE)

        state_batch = [data[0][-1] for data in mini_batch]
        action_batch = [data[1] for data in mini_batch]
        reward_batch = [data[2] for data in mini_batch]
        next_state_batch = [data[3][-1] for data in mini_batch]
        y_batch = []
        q_value_batch = self.q_value_t.eval(
            feed_dict={
                self.state_input_t: np.expand_dims(next_state_batch, axis=-1),
                self.rnn_seq_length: len(next_state_batch),
            }
        )
        for i in range(0, BATCH_SIZE):
            y_batch.append(reward_batch[i] + GAMMA * np.max(q_value_batch[i]))

        # train_concern = self.train_step.run(feed_dict={
        #     self.y_input: y_batch,
        #     self.action_input: action_batch,
        #     self.state_input: np.expand_dims(state_batch, axis=-1),
        # })
        train_concern = self.session.run(
           [self.train_step, self.cost],
           feed_dict={
               self.y_input: y_batch,
               self.action_input: action_batch,
               self.state_input: np.expand_dims(state_batch, axis=-1),
               self.rnn_seq_length: len(state_batch),
           }
        )
        # for not used
        train_concern
        if self.time_step in [500, 1000, 1500, 2000, 2500]:
            self.saver.save(self.session,
                            "saved_networks/" + "network" + "-dqn",
                            global_step=self.time_step)

        if self.time_step % self.UPDATE_TIME == 0:
            self.copy_target_q_network()

        # plt.scatter(self.time_step, train_concern[1],
        #             s=30, c='blue', marker='x')
        # plt.draw()
        # plt.pause(0.05)

    def make_action(self):
        # obtain current q_value
        # XXX I'm here until you check it!
        # print('xxx', len(self.current_state))
        q_value = self.q_value.eval(
            feed_dict={
                self.state_input: np.expand_dims(
                    self.current_state,
                    axis=-1
                ),
                self.rnn_seq_length: len(self.current_state),
            }
        )
        q_value = q_value[-1]
        # print(q_value)
        actions = np.zeros(self.action_nums)
        action_index = 0
        if random.random() <= self.epsilon:
            action_index = random.randrange(self.action_nums)
            actions[action_index] = 1
        else:
            action_index = np.argmax(q_value)
            actions[action_index] = 1

        if self.epsilon > FINAL_EPSILON and self.time_step > self.OBSERVE:
            # self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / self.EXPLORE
            if self.time_step % self.average_stage == 0:
                self.epsilon -= (
                    self.average_stage * (
                        INITIAL_EPSILON - FINAL_EPSILON) / self.EXPLORE
                )
        return actions, q_value

    def set_init_state(self, max_length):
        self.current_state = deque(
            (np.zeros((1, 100, 60))).tolist(),
            maxlen=max_length
        )
        # self.pre_doc_cursor = 0

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    # valid or same? padding before?
    def conv2d(self, x, w, strides=[1, 1, 1, 1]):
        return tf.nn.conv2d(x, w, strides=strides, padding="VALID")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 1, 1, 1],
                              padding="VALID")

    def max_pool_2x1(self, x):
        return tf.nn.max_pool(x,
                              ksize=[1, 2, 1, 1],
                              strides=[1, 1, 1, 1],
                              padding="VALID")

    # def reset_save_state(self):
    #     self.replay_memory = deque()
    def RNN(self, X, reuse, rnn_seq_length):
        rnn_inputs = 98*1*64
        rnn_hidden_units = 64  # TODO
        rnn_classes = 98*1*64
        # rnn_steps = 98
        # Define weights
        weights = {
            'in': tf.Variable(tf.random_normal([rnn_inputs, rnn_hidden_units])),
            'out': tf.Variable(
                tf.random_normal([rnn_hidden_units, rnn_classes])
            )
        }
        biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[rnn_hidden_units, ])),
            'out': tf.Variable(tf.constant(0.1, shape=[rnn_classes, ]))
        }

        X_in = tf.matmul(X, weights['in']) + biases['in']
        # X_in = tf.reshape(X_in, [-1, rnn_steps, rnn_hidden_units])

        cell = tf.contrib.rnn.BasicLSTMCell(rnn_hidden_units, reuse=reuse)
        init_state = cell.zero_state(BATCH_SIZE, dtype=tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(
            cell, X_in, initial_state=init_state,
            time_major=False, sequence_length=rnn_seq_length
        )

        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
        results = tf.matmul(outputs[-1], weights['out']) + biases['out']

        return results

    def get_q_values(self, docs):
        array_docs = np.array(docs, dtype=np.float)
        q_values = self.q_value.eval(
            feed_dict={
                self.state_input: np.expand_dims(
                    array_docs,
                    axis=-1
                ),
                self.rnn_seq_length: len(array_docs),
            }
        )
        return q_values
