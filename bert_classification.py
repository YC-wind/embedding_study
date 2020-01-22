#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-04-17 20:58
"""
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
from bert import modeling, tokenization, optimization
import pandas as pd
import collections
import os, json, time

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# init_checkpoint = "output/model.ckpt-87000"
init_checkpoint = "../bert/bert_model.ckpt"  # '../output/bert_topic.ckpt'
train_file = "./train.tf_record"
dev_file = "./dev.tf_record"
learning_rate = 5e-5
num_train_epochs = 30.0
train_batch_size = 32
eval_batch_size = 32
predict_batch_size = 8
warmup_proportion = 0.1
save_checkpoints_steps = 1000
iterations_per_loop = 1000
output_dir = "output"
max_seq_length = 128
use_tpu = False

flags = tf.flags
FLAGS = flags.FLAGS

bert_config_ = modeling.BertConfig.from_json_file("../bert/bert_config.json")
label2id = json.loads(open("./label2id.json").read())
label_list = list(label2id.keys())
tokenizer = tokenization.FullTokenizer(vocab_file="../bert/vocab.txt")


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def process_one_example(text_a, text_b=None):
    """
    :param text_a:
    :param text_b:
    :return:
    """
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = None
    if text_b:
        tokens_b = tokenizer.tokenize(text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    feature = (input_ids, input_mask, segment_ids)
    return feature


def prepare_tf_record_data(path="./train.csv", out_path="./train.tf_record"):
    """
        生成训练数据， tf.record, 多标签分类模型 train.csv : 0-text, 1-label_list
    :return:
    """
    df = pd.read_csv(path, index_col=0)
    df = shuffle(df)
    num_labels = len(label2id)
    writer = tf.python_io.TFRecordWriter(out_path)
    for index, row in df.iterrows():
        # label = label2id[row["topic"].strip()]
        t = row[1].strip()
        t_index = [label2id.get(_.strip()) for _ in t.split("###") if _ != ""]
        y = np.zeros(shape=(len(label2id)), dtype=np.int)
        try:
            y[t_index] = 1
        except:
            print(t, t_index)
            pass
        feature = process_one_example(row[0])

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()

        features["input_ids"] = create_int_feature(feature[0])
        features["input_mask"] = create_int_feature(feature[1])
        features["segment_ids"] = create_int_feature(feature[2])
        features["label_ids"] = create_int_feature(y)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

        if index % 1000 == 0:
            print(index)
    writer.close()


def get_input_data(input_file, seq_length, batch_size, num_labels):
    def parser(record):
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([num_labels], tf.int64),
            # "is_real_example": tf.FixedLenFeature([], tf.int64),
        }

        example = tf.parse_single_example(record, features=name_to_features)
        input_ids = example["input_ids"]
        input_mask = example["input_mask"]
        segment_ids = example["segment_ids"]
        labels = example["label_ids"]
        return input_ids, input_mask, segment_ids, labels

    dataset = tf.data.TFRecordDataset(input_file)
    dataset = dataset.map(parser).repeat().batch(batch_size).shuffle(buffer_size=1000)
    iterator = dataset.make_one_shot_iterator()
    input_ids, input_mask, segment_ids, labels = iterator.get_next()
    return input_ids, input_mask, segment_ids, labels


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value
    print(output_layer.shape)

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        # 修改为多标签的损失
        sigmoid_ = tf.nn.sigmoid(logits, name="logits")
        predictions = tf.cast((sigmoid_ > 0.5), tf.int32, name="predictions")

        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, tf.float32))
        loss = tf.reduce_mean(losses)

        # probabilities = tf.nn.softmax(logits, axis=-1)
        #         # log_probs = tf.nn.log_softmax(logits, axis=-1)
        #         #
        #         # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        #         #
        #         # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        #         # loss = tf.reduce_mean(per_example_loss)

        return (loss, sigmoid_, predictions)


def train():
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.gfile.MakeDirs(output_dir)

    train_examples_len = 4300
    dev_examples_len = 535
    num_labels = len(label2id)
    print(num_labels)
    num_train_steps = int(train_examples_len / train_batch_size * num_train_epochs)

    num_dev_steps = int(dev_examples_len / eval_batch_size)

    is_training = True

    seq_len = max_seq_length
    input_ids = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_ids')
    input_mask = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_mask')
    segment_ids = tf.placeholder(tf.int64, shape=[None, seq_len], name='segment_ids')
    labels = tf.placeholder(tf.int64, shape=[None, num_labels], name='labels')
    # labels = tf.placeholder(tf.int64, shape=[None], name='labels')
    use_one_hot_embeddings = False

    loss, sigmoid_, predictions = create_model(bert_config_, is_training, input_ids, input_mask,
                                               segment_ids, labels, num_labels,
                                               use_one_hot_embeddings)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    def metric_fn(label_ids, logits):
        predictions = np.argmax(logits, axis=-1)
        acc = np.sum(np.equal(label_ids, predictions)) / len(label_ids)
        # classification_report(label_ids, predictions)
        return acc

    def calculate_prf(y_pre, y_true):
        """

        :param y_pre:
        :param y_true:
        :return:
        """
        A = 0
        B = 0
        C = 0
        # 许多0的计算就没必要啦
        for p, y in zip(y_pre, y_true):
            p_l = [index for index, value in enumerate(p) if value == 1.0]
            y_l = [index for index, value in enumerate(y) if value == 1.0]
            comm = [i for i in p_l if i in y_l]
            A += len(comm)
            B += len(p_l)
            C += len(y_l)
        return A, B, C

    batch_size = train_batch_size
    input_ids2, input_mask2, segment_ids2, labels2 = get_input_data(train_file, seq_len, batch_size, num_labels)

    val_batch_size = eval_batch_size
    val_input_ids2, val_input_mask2, val_segment_ids2, val_labels2 = get_input_data(dev_file, seq_len,
                                                                                    val_batch_size, num_labels)

    init_global = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
    # saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_global)

        # 加载 参数
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                # var.trainable = False
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        # if 1:
        # latest_checkpoint = tf.train.latest_checkpoint('../output')

        # saver.restore(sess, latest_checkpoint)
        # print("checkpoint restored from %s" % latest_checkpoint)

        # tf.summary.FileWriter("output/",sess.graph)
        def train_step(ids, mask, segment, y):
            feed = {input_ids: ids,
                    input_mask: mask,
                    segment_ids: segment,
                    labels: y}
            _, out_loss, out_logits, p_ = sess.run([optimizer, loss, sigmoid_, predictions], feed_dict=feed)
            A, B, C = calculate_prf(p_, y)
            p = A / B if B > 0 else 0.0
            r = A / C if C > 0 else 0.0
            f = 2 * A / (B + C) if (B + C) > 0 else 0.0
            print("loss :{}, p :{}, r :{}, f :{}".format(out_loss, p, r, f))
            return out_loss, A, B, C

        def dev_step(ids, mask, segment, y):
            feed = {input_ids: ids,
                    input_mask: mask,
                    segment_ids: segment,
                    labels: y_train}
            out_loss, out_logits, p_ = sess.run([loss, sigmoid_, predictions], feed_dict=feed)
            A, B, C = calculate_prf(p_, y)
            p = A / B if B > 0 else 0.0
            r = A / C if C > 0 else 0.0
            f = 2 * A / (B + C) if (B + C) > 0 else 0.0
            print("+dev+loss :{}, p :{}, r :{}, f :{}".format(out_loss, p, r, f))
            return out_loss, A, B, C

        max_f = 0
        for i in range(num_train_steps):
            # batch 数据
            ids_train, mask_train, segment_train, y_train = sess.run([input_ids2, input_mask2, segment_ids2, labels2])
            print("step:", i, )
            train_step(ids_train, mask_train, segment_train, y_train)

            if i % 50 == 0:
                dev_total_loss = 0
                total_A = 0
                total_B = 0
                total_C = 0
                for j in range(num_dev_steps):  # 一个 epoch 的 轮数
                    ids_dev, mask_dev, segment_dev, y_dev = sess.run(
                        [val_input_ids2, val_input_mask2, val_segment_ids2, val_labels2])
                    print("step:", i, )
                    out_loss, A, B, C = dev_step(ids_dev, mask_dev, segment_dev, y_dev)
                    dev_total_loss += out_loss
                    total_A += A
                    total_B += B
                    total_C += C
                p = total_A / total_B
                r = total_A / total_C
                f = 2 * total_A / (total_C + total_B)
                print("P：", p)
                print("R：", r)
                print("F：", f)
                print("best:", max_f)
                if max_f < f:
                    print("save model:\t%f\t>%f" % (max_f, f))
                    max_f = f
                    saver.save(sess, '../output/bert_topic.ckpt', global_step=i)
        sess.close()


# prepare_tf_record_data(path="./train.csv", out_path="./train.tf_record")
# prepare_tf_record_data(path="./dev.csv", out_path="./dev.tf_record")

train()
