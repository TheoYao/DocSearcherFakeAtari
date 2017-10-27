# endcoding: utf-8

import tensorflow as tf
import numpy as np
import random
from collections import deque


INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
OBSERVE = 500.
EXPLORE = 1000.
REPLAY_MEMORY = 1000000
GAMMA = 0.95
BATCH_SIZE = 10
UPDATE_TIME = 100


class NeuralDQN:
    def __init__(self, action_nums=2):
        self.replay_memory = deque()

        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.action_nums = action_nums

        (self.state_input, self.q_value,
         self.w_conv1, self.b_conv1,
         self.w_conv2, self.b_conv2,
         self.w_fc1, self.b_fc1) = self.create_q_network()

        (self.state_input_t, self.q_value_t,
         self.w_conv1_t, self.b_conv1_t,
         self.w_conv2_t, self.b_conv2_t,
         self.w_fc1_t, self.b_fc1_t) = self.create_q_network()

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
        # self.session.run(tf.initialize_all_variables())
        self.session.run(tf.global_variables_initializer())
        checkoutpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkoutpoint and checkoutpoint.model_checkpoint_path:
            self.saver.restore(self.session,
                               checkoutpoint.model_checkpoint_path)
            print("Success loaded:", checkoutpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def create_q_network(self):
        state_in = tf.placeholder("float", [None, 100, 60, 1])
        state_input = tf.reshape(state_in, [-1, 100, 60, 1])

        w_conv1 = self.weight_variable([1, 30, 1, 32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(state_input, w_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        w_conv2 = self.weight_variable([1, 30, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, w_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x1(h_conv2)

        h_pool2_flat = tf.reshape(h_pool2, [-1, 98 * 1 * 64])

        w_fc1 = self.weight_variable([98 * 1 * 64, 1024])
        b_fc1 = self.bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

        w_fc2 = self.weight_variable([1024, self.action_nums])
        b_fc2 = self.bias_variable([self.action_nums])

        q_value = tf.matmul(h_fc1, w_fc2) + b_fc2

        return (state_input, q_value, w_conv1, b_conv1,
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

    def set_perception(self, next_observation, action,
                       reward, terminal, doc_cursor):
        new_state = next_observation

        # self.replay_memory.append(
        #     (self.current_state, action, reward,
        #      new_state, terminal, doc_cursor)
        # )

        self.replay_memory.append(
            (self.current_state[-1], action, reward,
             new_state[-1], terminal, doc_cursor)
        )

        if len(self.replay_memory) > REPLAY_MEMORY:
            self.replay_memory.popleft()
        if self.time_step > OBSERVE:
            self.train_q_network()

        state = ""
        if self.time_step <= OBSERVE:
            state = "observe"
        elif self.time_step > OBSERVE and self.time_step <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", self.time_step, "/ STATE", state,
              "/ EPSILON", self.epsilon)
        self.current_state = new_state
        self.time_step += 1

    def train_q_network(self):
        mini_batch = random.sample(self.replay_memory, BATCH_SIZE)
        # state_batch = [data[0][data[5]] for data in mini_batch]
        state_batch = [data[0] for data in mini_batch]

        action_batch = [data[1] for data in mini_batch]
        reward_batch = [data[2] for data in mini_batch]
        # next_state_batch = [data[3][data[5]] for data in mini_batch]
        next_state_batch = [data[3] for data in mini_batch]

        y_batch = []
        q_value_batch = self.q_value_t.eval(
            feed_dict={
                self.state_input_t: np.expand_dims(next_state_batch, axis=-1)
            }
        )
        for i in range(0, BATCH_SIZE):
            terminal = mini_batch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(
                    reward_batch[i] + GAMMA * np.max(q_value_batch[i])
                )

        self.train_step.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: np.expand_dims(state_batch, axis=-1),
        })

        if self.time_step % 1000 == 0:
            self.saver.save(self.session,
                            "saved_networks/" + "network" + "-dqn",
                            global_step=self.time_step)

        if self.time_step % UPDATE_TIME == 0:
            self.copy_target_q_network()

        if terminal:
            # self.reset_save_state()
            pass

    def make_action(self, doc_cursor):
        # obtain current q_value
        q_value = self.q_value.eval(
            feed_dict={
                self.state_input: np.expand_dims(self.current_state, axis=-1)
            }
        )[0]
        actions = np.zeros(self.action_nums)
        action_index = 0
        if random.random() <= self.epsilon:
            action_index = random.randrange(self.action_nums)
            actions[action_index] = 1
        else:
            action_index = np.argmax(q_value)
            actions[action_index] = 1

        if self.epsilon > FINAL_EPSILON and self.time_step > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return actions

    def set_init_state(self, observation):
        self.current_state = np.stack(
            # (np.expand_dims(observation, axis=-1)),
            (observation),
            axis=0
        )

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

    def reset_save_state(self):
        self.replay_memory = deque()
