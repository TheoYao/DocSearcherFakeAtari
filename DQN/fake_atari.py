# encoding: utf-8
import pickle
import sys
from collections import deque
import matplotlib.pyplot as plt


class DocSeq:
    def __init__(self, aim_amount):
        self.documents = deque(maxlen=aim_amount)
        self.candi_docs = []
        self.candi_cursor = 0

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
        if is_pos:
            reward = 1 if action_str == 'choose' else 0
        else:
            reward = -1 if action_str == 'choose' else 0
        if action_str == 'choose':
            self.documents.append(doc)
        self.candi_cursor += 1
        if self.candi_cursor == len(self.candi_docs):
            plt.ioff()
            plt.show()
            sys.exit(0)

        return self.documents, reward  # self.is_over, self.doc_cursor-1

    def load_doc(self):
        data = pickle.load(
            open("../Data/total.pkl", "rb")
        )
        positive_docs = data["positive"][:7000]
        negative_docs = data["negative"][:7000]
        self.candi_docs = [item for sublist in zip(
            negative_docs, positive_docs
        ) for item in sublist]
