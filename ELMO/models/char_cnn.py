#!/usr/bin/python
# coding:utf-8
"""
@author: Cong Yu

@time: 2018/12/3 11:33 AM
"""
import tensorflow as tf


class CharCNNEmbedding:
    def __init__(self, config):
        self.char_vocab_size = config["char_vocab_size"]
        self.char_embedding_dim = config["char_embedding_dim"]

        self.kernel_sizes = config["kernel_sizes"]
        self.filter_size = config["elmo_hidden"] // len(self.kernel_sizes)

        self.seq_len = config["word_seq_len"]
        self.char_seq_len = config["char_seq_len"]

        with tf.variable_scope("char_cnn", reuse=tf.AUTO_REUSE):
            self.conv_filters = [
                tf.layers.Conv1D(self.filter_size, kernel_size)
                for kernel_size in self.kernel_sizes
            ]

        with tf.variable_scope("char_embedding", reuse=tf.AUTO_REUSE):
            self.embedding_weight = tf.get_variable("embedding_weight", [self.char_vocab_size, self.char_embedding_dim],
                                                    dtype=tf.float32)

    def forward(self, data):
        # batch * word * char * char_e
        embed_input = tf.nn.embedding_lookup(self.embedding_weight, data["input"])

        conv_outputs = []
        # (batch * word) * char * char_e
        conv_input = tf.reshape(embed_input, [-1, self.char_seq_len, self.char_embedding_dim])
        for conv, kernel_size in zip(self.conv_filters, self.kernel_sizes):
            # (batch * word) * char * filter_size
            conv_output = conv(conv_input)
            # batch * word * char * filter_size
            _conv_output = tf.reshape(conv_output, [-1, self.seq_len, conv_output.shape[1], self.filter_size])
            # batch * word * filter_size
            pool_output = tf.nn.max_pool(_conv_output, [1, 1, conv_output.shape[1], 1], [1, 1, 1, 1], 'VALID')
            pool_output = tf.squeeze(pool_output, axis=2)
            conv_outputs.append(pool_output)

        # shape = (batch_size, seq_len, embedding_dim)
        char_word_embedding = tf.concat(conv_outputs, axis=2)
        return char_word_embedding
