#!/usr/bin/python
# coding:utf-8
"""
@author: Cong Yu

@time: 2018/12/3 11:33 AM
"""
import tensorflow as tf
from models.char_cnn import CharCNNEmbedding


class ELMO:
    def __init__(self, config):
        self.embedding = CharCNNEmbedding(config)
        self.hidden_size = config["elmo_hidden"]
        self.vocab_size = config["word_vocab_size"]
        self.seq_len = config["word_seq_len"]
        self.config = config

        with tf.variable_scope("elmo_rnn_cell"):
            self.forward_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, reuse=tf.AUTO_REUSE)
            self.backward_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, reuse=tf.AUTO_REUSE)

        if config.get("use_skip_connection"):
            self.forward_cell = tf.nn.rnn_cell.ResidualWrapper(self.forward_cell)
            self.backward_cell = tf.nn.rnn_cell.ResidualWrapper(self.backward_cell)

        with tf.variable_scope("elmo_softmax"):
            softmax_weight_shape = [config["word_vocab_size"], config["elmo_hidden"]]

            self.forward_softmax_w = tf.get_variable("forward_softmax_w", softmax_weight_shape, dtype=tf.float32)
            self.backward_softmax_w = tf.get_variable("backward_softmax_w", softmax_weight_shape, dtype=tf.float32)

            self.forward_softmax_b = tf.get_variable("forward_softmax_b", [config["word_vocab_size"]])
            self.backward_softmax_b = tf.get_variable("backward_softmax_b", [config["word_vocab_size"]])

    def forward(self, data):
        # shape = (batch_size, seq_len, embedding_dim)
        embedding_output = self.embedding.forward(data)
        with tf.variable_scope("elmo_rnn_forward"):
            # shape = (batch_size, seq_len, hidden_size)
            forward_outputs, forward_states = tf.nn.dynamic_rnn(self.forward_cell,
                                                                inputs=embedding_output,
                                                                sequence_length=data["input_len"],
                                                                dtype=tf.float32)

        with tf.variable_scope("elmo_rnn_backward"):
            # shape = (batch_size, seq_len, hidden_size)
            backward_outputs, backward_states = tf.nn.dynamic_rnn(self.backward_cell,
                                                                  inputs=embedding_output,
                                                                  sequence_length=data["input_len"],
                                                                  dtype=tf.float32)

        # # Concatenate the forward and backward LSTM output
        forward_projection = tf.matmul(forward_outputs, tf.expand_dims(tf.transpose(self.forward_softmax_w), 0))
        forward_projection = tf.nn.bias_add(forward_projection, self.forward_softmax_b)

        backward_projection = tf.matmul(backward_outputs, tf.expand_dims(tf.transpose(self.backward_softmax_w), 0))
        backward_projection = tf.nn.bias_add(backward_projection, self.backward_softmax_b)
        # shape = (batch_size, seq_len, elmo_hidden)
        return forward_outputs, backward_outputs, forward_projection, backward_projection

    def train(self, data, global_step_variable=None):
        forward_output, backward_output, forward_projection, backward_projection = self.forward(data)

        forward_target = data["target"]
        # shape = (batch_size, seq_len)
        forward_pred = tf.cast(tf.argmax(tf.nn.softmax(forward_projection, -1), -1), tf.int32)
        # 类似 auto encoder， 输入是结构化 word_char, 输出是 word
        forward_correct = tf.equal(forward_pred, forward_target)
        forward_padding = tf.sequence_mask(data["target_len"], maxlen=self.seq_len, dtype=tf.float32)

        forward_softmax_target = tf.cast(tf.reshape(forward_target, [-1, 1]), tf.int64)
        # shape = (batch_size, （seq_len* elmo_hidden)
        forward_softmax_input = tf.reshape(forward_output, [-1, self.hidden_size])
        forward_train_loss = tf.nn.sampled_softmax_loss(
            weights=self.forward_softmax_w, biases=self.forward_softmax_b,
            labels=forward_softmax_target, inputs=forward_softmax_input,
            num_sampled=self.config["softmax_sample_size"],
            num_classes=self.config["word_vocab_size"]
        )

        forward_train_loss = tf.reshape(forward_train_loss, [-1, self.seq_len])
        forward_train_loss = tf.multiply(forward_train_loss, forward_padding)
        forward_train_loss = tf.reduce_mean(forward_train_loss)

        backward_target = tf.reverse_sequence(data["target"], data["target_len"], seq_axis=1, batch_axis=0)
        backward_pred = tf.cast(tf.argmax(tf.nn.softmax(backward_projection, -1), -1), tf.int32)
        backward_correct = tf.equal(backward_pred, backward_target)
        backward_padding = tf.sequence_mask(data["target_len"], maxlen=self.seq_len, dtype=tf.float32)

        backward_softmax_target = tf.cast(tf.reshape(backward_target, [-1, 1]), tf.int64)
        backward_softmax_input = tf.reshape(backward_output, [-1, self.hidden_size])
        backward_train_loss = tf.nn.sampled_softmax_loss(
            weights=self.backward_softmax_w, biases=self.backward_softmax_b,
            labels=backward_softmax_target, inputs=backward_softmax_input,
            num_sampled=self.config["softmax_sample_size"],
            num_classes=self.config["word_vocab_size"]
        )

        backward_train_loss = tf.reshape(backward_train_loss, [-1, self.seq_len])
        backward_train_loss = tf.multiply(backward_train_loss, backward_padding)
        backward_train_loss = tf.reduce_mean(backward_train_loss)

        train_loss = forward_train_loss + backward_train_loss
        train_correct = tf.concat([forward_correct, backward_correct], axis=-1)
        train_acc = tf.reduce_mean(tf.cast(train_correct, tf.float32))

        tf.summary.scalar("train_acc", train_acc)
        tf.summary.scalar("train_loss", train_loss)

        train_ops = tf.train.AdamOptimizer().minimize(train_loss)
        return train_loss, train_acc, train_ops

    def pred(self, data):
        elmo_projection_output = self.forward(data)
        eval_output = tf.nn.softmax(elmo_projection_output, dim=-1)
        return eval_output
