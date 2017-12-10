# encoding: utf-8
import pickle
import sys
from collections import deque
# import matplotlib.pyplot as plt


class DocSeq:
    def __init__(self, aim_amount):
        self.documents = deque(maxlen=aim_amount)
        self.candi_docs = []
        self.candi_cursor = 0
        self.train_test_boundary = 7000
        self.test_docs = []
        self.test_cursor = 0
        self.load_doc()

    def get_cur_seq(self):
        return self.document

    def reset(self):
        pass

    def next(self, action):
        doc = self.candi_docs[self.candi_cursor][1]
        is_pos = self.candi_docs[self.candi_cursor][0]
        if action[0] > 0:
            action_str = 'choose'
        else:
            action_str = 'skip'
        reward = 0
        is_optimal = 0
        if is_pos:
            reward = 1 if action_str == 'choose' else 0
            is_optimal = reward
        else:
            reward = -1 if action_str == 'choose' else 0
            is_optimal = 1 if reward == -1 else 0
        if action_str == 'choose':
            self.documents.append(doc)
        self.candi_cursor += 1
        if self.candi_cursor >= len(self.candi_docs):
            # plt.ioff()
            # plt.show()
            return None, None, None
        return self.documents, reward, is_optimal

    def get_test_document(self):
        if self.test_cursor == len(self.test_docs):
            sys.exit(0)
        self.test_cursor += 1
        return [self.test_docs[self.test_cursor-1]]

    def load_doc(self):
        data = pickle.load(
            open("../Data/total.pkl", "rb")
        )
        # XXX
        positive_docs = data["positive"][:self.train_test_boundary]
        negative_docs = data["negative"][:self.train_test_boundary]
        self.candi_docs = [item for sublist in zip(
            negative_docs, positive_docs
        ) for item in sublist]

        # test_positive_docs = data["positive"][self.train_test_boundary:]
        # test_negative_docs = data["negative"][self.train_test_boundary:]
        # self.test_docs = [item for sublist in zip(
        #     test_negative_docs, test_positive_docs
        # ) for item in sublist]
