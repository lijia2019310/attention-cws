# attention-cws

本人硕士学位论文中基于纯注意力机制的中文分词模型的Keras实现。

## src
- 其中单层网络实现为cws.py，多层多元组注意力网络为ngrams.py，可直接使用命令行参数进行训练、预测、调参等；
- params.py可直接自动进行一维的最优超参数搜索，attention.py参考[博客](https://kexue.fm/archives/4765)构建Transformer中注意力单元；

## icwb2
- Bakeoff 2005公开提供的四种数据集；

论文pdf版已附在主目录下