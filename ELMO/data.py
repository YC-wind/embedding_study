#!/usr/bin/python
# coding:utf-8
"""
@author: Cong Yu

@time: 2018/12/20 4:41 PM
"""
import os
from collections import OrderedDict, Counter


class Vocab():
    def __init__(self, voc_path, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        print("Building Vocab")
        self.itos = list(["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"])
        max_size = None if max_size is None else max_size + len(self.itos)
        # voc 文件已经生成好了，直接读
        for line in open(voc_path).readlines():
            if line.strip() == "":
                continue
            self.itos.append(line.strip().split()[0])
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def to_seq(self, sentence: str, seq_len: int = None, with_eos=False, with_sos=False, with_len=False,
               mid_pad=False) -> list:
        tokens = self.tokenizer(sentence)
        seq = [self.stoi.get(c, self.unk_index) for c in tokens]

        if with_eos:
            seq += [self.eos_index]  # this would be index 1
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            if not mid_pad:
                seq += [self.pad_index for _ in range(seq_len - len(seq))]
            else:
                front_pad = [self.pad_index for _ in range(int((seq_len - len(seq)) / 2))]
                end_path = [self.pad_index for _ in range(seq_len - len(seq) - len(front_pad))]
                seq = front_pad + seq + end_path
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        tokens = [self.itos[idx]
                  if idx < len(self.itos)
                  else "<%d>" % idx
                  for idx in seq
                  if with_pad or idx != self.pad_index]

        return self.joiner(tokens) if join else tokens

    def tokenizer(self, sentence: str) -> list:
        return sentence.strip().split()

    def joiner(self, tokens: list) -> str:
        return " ".join(tokens)

    def __len__(self):
        return len(self.itos)


class CharWordVocab(Vocab):
    def to_seq(self, sentence: str, seq_len: int = None,
               char_seq_len=10, word_seq_len=10, mid_pad=True,
               with_eos=False, with_sos=False, with_len=False):
        seq = [
            super(CharWordVocab, self).to_seq(token, seq_len=char_seq_len, mid_pad=mid_pad)
            for token in sentence.split()
        ]

        if with_eos:
            seq += [self.eos_index]  # this would be index 1
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [[self.pad_index for _ in range(char_seq_len)]
                    for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        tokens = [super(CharWordVocab, self).from_seq(token, join=True, with_pad=False)
                  for token in seq]
        return " ".join(tokens) if join else tokens





class Vocabulary():
    """
        数据集 搭建词典，字典的过程
    """

    def __init__(self, config):
        # 最小 词语 出现次数
        self.mini_count_word = config["mini_count_word"]
        # 最小 字 出现次数
        self.mini_count_char = config["mini_count_char"]
        # 一句话 最大 词语 长度
        self.word_seq_len = config["word_seq_len"]
        # 一个词 最大 字 长度
        self.char_seq_len = config["char_seq_len"]
        # 使用 1，2，3分别代表 起始、结束、未登录词
        self.bos = 1
        self.eos = 2
        self.unk = 3

    def prepare_word_dict(self, seg_path):
        """
            读取训练文件，生成词典并按 词频 进行降序排序
        :param seg_path:        已经分好词的文件路径
        :return:
        """
        word2count = OrderedDict()
        # "< S >, < / S >, < UNK >", 添加 起始、结束、未登录词 的标志
        for file in os.listdir(seg_path):
            for line in open(seg_path + file).readlines():
                for word in line.strip().split():
                    temp_w = word2count.get(word)
                    if temp_w:
                        word2count[word] += 1
                    else:
                        word2count[word] = 1
        # 遍历完整个数据集后，进行 降序 排序
        word2count_new = sorted(word2count.items(), key=lambda k: k[1], reverse=True)
        with open("./result/vocabulary_word.txt", "w") as fw:
            fw.write("<S>" + "\t" + "1" + "\n")
            fw.write("</S>" + "\t" + "2" + "\n")
            fw.write("<UNK>" + "\t" + "3" + "\n")
            for _ in word2count_new:
                if _[1] >= self.mini_count_word:
                    fw.write(_[0] + "\t" + str(_[1]) + "\n")
                else:
                    break

    def prepare_char_dict(self, seg_path):
        """
            读取训练文件，生成字典并按 字频 进行降序排序
        :param seg_path:        已经分好词的文件路径
        :return:
        """
        char2count = OrderedDict()
        # "< S >, < / S >, < UNK >", 添加 起始、结束、未登录词 的标志
        for file in os.listdir(seg_path):
            for line in open(seg_path + file).readlines():
                for word in line.strip().split():
                    for c in word:
                        temp_c = char2count.get(c)
                        if temp_c:
                            char2count[c] += 1
                        else:
                            char2count[c] = 1
        # 遍历完整个数据集后，进行 降序 排序
        char2count_new = sorted(char2count.items(), key=lambda k: k[1], reverse=True)
        with open("./result/vocabulary_char.txt", "w") as fw:
            for _ in char2count_new:
                if _[1] >= self.mini_count_char:
                    fw.write(_[0] + "\t" + str(_[1]) + "\n")
                else:
                    break

    def prepare_word_char_dict(self, seg_path):
        """
            生成 词典、字典；起始，结束、未登录词不包含在内
        :param seg_path:        已经分好词的文件路径
        :return:
        """
        word2count = OrderedDict()
        char2count = OrderedDict()
        # "< S >, < / S >, < UNK >", 添加 起始、结束、未登录词 的标志
        for file in os.listdir(seg_path):
            for line in open(seg_path + file).readlines():
                for word in line.strip().split():
                    # 词典的词频统计
                    temp_w = word2count.get(word)
                    if temp_w:
                        word2count[word] += 1
                    else:
                        word2count[word] = 1
                    for c in word:
                        # 字典的词频统计
                        temp_c = char2count.get(c)
                        if temp_c:
                            char2count[c] += 1
                        else:
                            char2count[c] = 1

        # 遍历完整个数据集后，进行 降序 排序
        word2count_new = sorted(word2count.items(), key=lambda k: k[1], reverse=True)
        with open("./result/vocabulary_word.txt", "w") as fw:
            for _ in word2count_new:
                if _[1] >= self.mini_count_word:
                    fw.write(_[0] + "\t" + str(_[1]) + "\n")
                else:
                    break

        # 遍历完整个数据集后，进行 降序 排序
        char2count_new = sorted(char2count.items(), key=lambda k: k[1], reverse=True)
        with open("./result/vocabulary_char.txt", "w") as fw:
            for _ in char2count_new:
                if _[1] >= self.mini_count_char:
                    fw.write(_[0] + "\t" + str(_[1]) + "\n")
                else:
                    break


if __name__ == "__main__":
    config = {
        "mini_count_word": 5,
        "mini_count_char": 10,
        "word_seq_len": 128,
        "char_seq_len": 8,
    }
    voc = Vocabulary(config)
    voc.prepare_word_char_dict("./seg_data/")
