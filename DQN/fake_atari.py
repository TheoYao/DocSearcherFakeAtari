# encoding: utf-8
import numpy as np
import pickle
import sys


class DocSeq:
    def __init__(self, aim_amount):
        self.aim_amount = aim_amount
        self.documents = np.zeros((self.aim_amount, 100, 60))
        self.is_over = False
        self.candi_docs = []
        self.candi_cursor = 0
        self.doc_cursor = 0

        self.load_doc()

    def get_cur_seq(self):
        return self.document

    def reset(self):
        self.documents = np.zeros((self.aim_amount, 100, 60))
        self.doc_cursor = 0
        self.is_over = False
        # return self.get_cur_seq()

    def next(self, action):
        if self.is_over:
            self.reset()
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
            # np.append(self.documents, doc)
            # self.documents[:, :, self.doc_cursor] = doc
            self.documents[self.doc_cursor] = doc
            self.doc_cursor += 1
        if self.doc_cursor == self.aim_amount - 1:
            self.is_over = True
        self.candi_cursor += 1
        if self.candi_cursor == len(self.candi_docs):
            sys.exit(0)

        return self.documents, reward, self.is_over, self.doc_cursor-1

    def load_doc(self):
        data = pickle.load(
            open("../Data/total.pkl", "rb")
        )
        positive_docs = data["positive"]
        negative_docs = data["negative"]
        # positive_docs = list(map(lambda x: [True, x], positive_docs))
        # negative_docs = list(map(lambda x: [False, x], negative_docs))
        self.candi_docs = [item for sublist in zip(
            negative_docs, positive_docs
        ) for item in sublist]
