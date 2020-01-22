# ELMO-tf
> ELMO : Deep contextualized word representations 
> https://arxiv.org/abs/1802.05365

## 说明

参考 [ELMO-tf](https://github.com/codertimo/ELMO-tf),修改部分代码，适应于中文语料。
这位韩国小哥哥写的代码很清晰，相对于原始的实现，可读性好很多。原始的实现需要自行整理，搭建中文处理机制。

- 最原始的 ELMO 实现原理，请参考 [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)

- 其他基于 ELMO 的chinese 版本实现，请访问 [ELMO chinese](https://github.com/searobbersduck/ELMo_Chin).

## 使用
``python3.6`` ``tf>1.12.0`` 

具体的tf最低要求环境没有研究，本机tf=1.12

```bash
python3 main.py
```

char 级别的 ELMO 效果如下，字典在5000+，词典在7w+， 模型下来快一个g, blablabla。
模型是跑起来了，但是未测试使用效果，只当作学习。


```plaint text
{'batch_size': 1024, 'corpus_files': './seg_data/', 'epochs': 10, 'verbose_freq': 1, 'word_vocab_path': './result/vocabulary_word.txt', 'char_vocab_path': './result/vocabulary_char.txt', 'word_seq_len': 128, 'char_seq_len': 8, 'char_embedding_dim': 64, 'kernel_sizes': [1, 2, 3, 4], 'filter_sizes': None, 'elmo_hidden': 512, 'softmax_sample_size': 8196, 'prefetch_size': 1024, 'log_dir': './logs/', 'save_freq': 1000, 'model_save_path': './output/elmo.model.test', 'log_file_prefix': 'elmo.log'}
Building Vocab
Building Vocab
DataSet Size: 541889
Instructions for updating:
Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.
2018-12-21 10:22:17.786025: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
start training...
Train EP:0 [0/529] loss: 5.882568 acc: 0.730366
Train EP:0 [1/529] loss: 6.964141 acc: 0.680687
```

模型大致结构 为 

x = [w1,w2,w3,...wn,pad,pad...max_seq_len] # x的 word 组成是有 char 词典构成的 id 序列，加上了text cnn 部分。使用的是 char-embedding，即使用的是char——voc

**来预测**

y=[w2,w3,w4,...wn,eos,pad,pad...max_seq_len]# y的 word 组成是有 word 字典对应的 id。使用的是 char-embedding，即使用的是word——voc

后面在反向做以下，就是双向语言模型了。

在这样的结构下，中文语料训练会使模型的参数大很多，耗时较大。

## next

基于此结构，搭建 char 级别的 ELMO 字向量，相对参数会少很多。
有错误的地方，还请各位大佬指正。

