#!/usr/bin/python
# coding:utf-8
"""
@author: Cong Yu
@time: 2018/12/6 7:28 PM
"""
import os, time, sys
sys.path.append("../")
import tensorflow as tf
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from bert import modeling, tokenization
from bert.extract_features import model_fn_builder, convert_lst_to_features, PoolingStrategy

# 获取当前文件的上层路径
path = os.path.dirname(os.path.abspath(__file__))
model_dir = "/Users/yucong/PycharmProjects/helloAi/bert"
config_fp = os.path.join(model_dir, 'bert_config.json')
checkpoint_fp = os.path.join(model_dir, 'bert_model.ckpt')
vocab_fp = os.path.join(model_dir, 'vocab.txt')
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_fp)
max_seq_len = 10
worker_id = id
daemon = True
model_fn = model_fn_builder(
    bert_config=modeling.BertConfig.from_json_file(config_fp),
    init_checkpoint=checkpoint_fp,
    pooling_strategy=PoolingStrategy.NONE,
    pooling_layer=[-2]
)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
estimator = Estimator(model_fn, config=RunConfig(session_config=config), model_dir=None)


def input_fn_builder(msg):
    def gen():
        for i in range(1):
            tmp_f = list(convert_lst_to_features(msg, max_seq_len, tokenizer))
            yield {
                'input_ids': [f.input_ids for f in tmp_f],
                'input_mask': [f.input_mask for f in tmp_f],
                'input_type_ids': [f.input_type_ids for f in tmp_f]
            }

    def input_fn():
        for i in gen():
            print(i)
        return (tf.data.Dataset.from_generator(
            gen,
            output_types={'input_ids': tf.int32,
                          'input_mask': tf.int32,
                          'input_type_ids': tf.int32,
                          },
            output_shapes={
                'input_ids': (None, max_seq_len),
                'input_mask': (None, max_seq_len),
                'input_type_ids': (None, max_seq_len)}).prefetch(10))

    return input_fn


t1 = time.time()
input_fn = input_fn_builder(["NLP好难啊！", "怎么办呢？"])

result = estimator.predict(input_fn)
for rq in result:
    a = rq['encodes']
    print(rq['encodes'])
t2 = time.time()
print(a.shape)
print("cost time:", t2 - t1)
