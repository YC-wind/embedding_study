# bert

## 说明
参考 [bert-as-service](https://github.com/hanxiao/bert-as-service),只保留生成embedding的代码。

- 想要了解具体的bert实现，请参考 [google-bert](https://github.com/google-research/bert)

- 已经预训练好的模型，请访问 [模型链接](https://github.com/google-research/bert#pre-trained-models).

- bert_classification。基于 bert 的分类模型，下游任务（二分类、多分类、多标签分类都可以修改）

# ELMO-tf
> ELMO : Deep contextualized word representations 
> https://arxiv.org/abs/1802.05365
## 说明

参考 [ELMO-tf](https://github.com/codertimo/ELMO-tf),修改部分代码，适应于中文语料。
这位韩国小哥哥写的代码很清晰，相对于原始的实现，可读性好很多。原始的实现需要自行整理，搭建中文处理机制。

- 最原始的 ELMO 实现原理，请参考 [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)

- 其他基于 ELMO 的chinese 版本实现，请访问 [ELMO chinese](https://github.com/searobbersduck/ELMo_Chin).

# word2vec
> 中文词向量：https://github.com/Embedding/Chinese-Word-Vectors
> 腾讯词向量：链接:https://pan.baidu.com/s/1meeKUBKbGMyTGrx664F4Ng  密码:xfh1
## 说明

执行word2vec目录下，word2vec_embedding.py文件即可。

该项目使用腾讯词向量进行 词向量、句向量的计算。

- 词向量（查表，没有就按字）
- 句向量（词性加权，词向量，最后求平均）

使用：

下载腾讯词向量，tencent_45000.txt 应该就可以了


# 其他

```
#!/usr/bin/env bash

# 重置git的方法，（第二步需要 修改，git add 最好手动确定下）
#1. Checkout
git checkout --orphan latest_branch
#2. Add all the files
git add -A
#3. Commit the changes
git commit -am "commit message"
#4. Delete the branch
git branch -D master
#5.Rename the current branch to master
git branch -m master
#6.Finally, force update your repository
git push -f origin master
```







