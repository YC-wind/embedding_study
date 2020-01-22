#!/usr/bin/python
# coding:utf-8
"""
@author: Cong Yu

@time: 2018/12/3 11:33 AM
"""
import data,os

# 需要自己搭建数据集，处理 词典 与 字典

class ElmoChineseDataset:
    """
        中文 ELMO 模型 ，数据集
    """

    def __init__(self, config):
        self.corpus_files = config["corpus_files"]
        # self.jamo_processor = Han2Jamo()

        # self.char_vocab = CharWordVocab.load_vocab(config["char_vocab_path"])
        self.char_vocab = data.CharWordVocab(config["char_vocab_path"])
        # self.word_vocab = WordVocab.load_vocab(config["word_vocab_path"])
        self.word_vocab = data.Vocab(config["word_vocab_path"])

        self.seq_len = config["word_seq_len"]
        self.char_seq_len = config["char_seq_len"]
        self.corpus_size = self.get_corpus_size()
        print("DataSet Size:", self.corpus_size)

        config["char_vocab_size"] = len(self.char_vocab)
        config["word_vocab_size"] = len(self.word_vocab)

    def text_to_char_sequence(self, text):
        char_idx_seq, seq_len = self.char_vocab.to_seq(text,
                                                       char_seq_len=self.char_seq_len,
                                                       seq_len=self.seq_len,
                                                       with_len=True)
        seq_len = self.seq_len if seq_len > self.seq_len else seq_len
        return char_idx_seq, seq_len

    def text_to_word_sequence(self, text):
        """
            原始作者是 只后面添加 结束标志
        :param text:
        :return:
        """
        # 因为 with_eos, 已经补了 1个 标志符， 相当于 后移一位
        word_idx_seq, seq_len = self.word_vocab.to_seq(text, seq_len=self.seq_len + 1, with_len=True, with_eos=True)
        # 处理异常情况，seq——len 为实际长度（+1）
        seq_len = self.seq_len + 1 if seq_len > self.seq_len + 1 else seq_len
        word_idx_seq, seq_len = word_idx_seq[1:], seq_len - 1
        # seq——len 为真实长度（no1）模拟 （1，2，3，... n）+pad部分 -> (2,3,4...n,eos) +pad部分
        return word_idx_seq, seq_len

    def produce_data(self, text):
        text = text.strip()
        char_word_input, input_len = self.text_to_char_sequence(text)
        word_target, target_len = self.text_to_word_sequence(text)

        return {"input": char_word_input, "input_len": input_len,
                "target": word_target, "target_len": target_len}

    def data_generator(self):
        files = os.listdir(self.corpus_files)
        for file_path in files:
            with open(self.corpus_files + file_path, "r", encoding="utf-8") as f:
                for text in f.readlines():
                    if text.strip() == "":
                        continue
                    yield self.produce_data(text)

    def get_corpus_size(self):
        count = 0
        files = os.listdir(self.corpus_files)
        for file_path in files:
            with open(self.corpus_files + file_path) as file:
                for text in file.readlines():
                    if text.strip() == "":
                        continue
                    count += 1
        return count
