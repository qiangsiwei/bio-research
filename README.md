生物文本信息抽取
=============

整体框架
----

本文数据采集于scopus，包括论文摘要、正文以及其他相关信息，试图从论文的摘要及正文部分抽取生物领域多种实体，主要包括：疾病、基因、突变、通路、药物、代谢、蛋白等之间的相互作用关系。

整体框架如图，主要包括：数据采集、文本预处理、实体提取、关系提取、建立信息搜索引擎与知识图谱可视化。

![Alt Text](https://raw.githubusercontent.com/qiangsiwei/bio-research/master/figure/m1.png)

实体提取
----

实体提取采用双层双向循环神经元网络，相比于单层单向循环神经元网络，能够充分利用词序列词语两侧各节点的隐含层信息，达到更好的效果。训练过程中除了神经元模型学习，还同时对关键词表进行扩充，读写过程如图所示。

![Alt Text](https://raw.githubusercontent.com/qiangsiwei/bio-research/master/figure/m2.png)

关系提取
----

关系提取采用自监督学习方法，首先基于约束集搜索文本语法与依存树解析结果，得到可靠的正负样本，之后基于浅层文本特征训练分类模型，快速抽取实体间关系，过程如图所示。

![Alt Text](https://raw.githubusercontent.com/qiangsiwei/bio-research/master/figure/m3.png)
