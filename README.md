# <center>2022 DCIC 基于文本字符的交易验证码识别</center>

2022 DCIC 兴业银行验证码OCR识别 小小快乐 队伍方案及源码

## <center>1. 赛题背景 </center>
验证码作为性价较高的安全验证方法，在多场合得到了广泛的应用，有效地防止了机器人进行身份欺骗，其中，以基于文本字符的静态验证码最为常见。随着使用的深入，噪声点、噪声线、重叠、形变等干扰手段层出不穷，不断提升安全防范级别。RPA技术作为企业数字化转型的关键，因为其部署的非侵入式备受企业青睐，验证码识别率不高往往限制了RPA技术的应用。一个能同时过滤多种干扰的验证码模型，对于相关自动化技术的拓展使用有着一定的商业价值。

####建模方案：

建模方案主要尝试了三种：
- 第一种就是用OCR方法CTC loss, 具体代码见./ctc_pytorch/文件夹,尝试了OCR方法的一些数据增强操作及搭建模型; 
- 第二种则是利用 CrossEntropyLoss 多分类的方法, 具体代码见./pt_classify/, 尝试了cutmix,mixup,cutout,及常见的数据增强方法等; 
- 第三种是 MultiLabelSoftMarginLoss 的多标签多分类方法, 具体代码见./classify_248/, 结合第二种方案的调参经验主要尝试了 cutout, mixup 的数据增强方法, 之后调参优化模型泛化性能。

## <center>2.具体方案</center>
####2.1.文件夹结构
```
root 
    - CTC_model   CTC loss 方案代码
		- config 超参数配置
        - data   数据集存放路径
        - model  模型文件及预训练模型
            - ...
        - torchtools 会用到的库
        - utils  数据加载及数据增强
            - ...
        - weights 训练得到的模型保存路径
        Train_pt.py  训练模型
        Test_pt.py   预测模型

    - CrossEntropy_model   CrossEntropy loss 方案代码
        - config 超参数配置
        - data   数据集存放路径
            - ...
        - model  模型文件及预训练模型
            - ...
        - utils  
            - ...
        - weights 训练得到的模型保存路径
        Train_CrossEntropyLoss.py  训练模型
        predict.py                 预测模型

    - MultiLabelSoftMargin_model   MultiLabelSoftMargin loss 方案代码
        - config 超参数配置
        - data   数据集存放路径
            - ...
        - model  模型文件及预训练模型
            - ...
        - utils  
            - ...
        - weights 训练得到的模型保存路径
        Train_pt.py  训练模型
        predict.py   预测模型
    
    - Tool_code   一些工具代码,包括五折交叉训练集制作,伪标签制作等
        - captcha 其中一种验证码制作
        - Data    对数据集的处理操作
        - Img_aug 图像探索
```

####2.2.CTC loss 方案

**建模思路:** 
先是随便找了个模型训练了一遍, 发现过拟合严重, 之后不断调整网络结构, 最终得到一个500K的模型, 准确率达到了0.92出现了瓶颈, 无法继续提分, 之后就换了多标签分类方案, 现在回头看ctc loss的方案 如果加上cutout等图片分类中的数据增强方法, 效果应该会不错, 后续主要尝试分类方案了, 所以就没有回头再搞这个ctc方案。

**数据增强尝试:**
具体包括gauss模糊、norm模糊、锐化、滤波、随机添加干扰线(横线、竖线)、随机调整对比度、颜色随机扰动、随机拼接、随机裁剪、随机缩放、噪点、(扭曲、伸展、透镜)。
根据训练经验来看，这些数据增强操作有一定效果，但是效果并没有很大。

####2.3.CrossEntropy loss 方案
**建模思路:**

采用了 CrossEntropyLoss 分别计算每个字符的 loss, 尝试了EfficientNet,RepVGG等系列模型,数据增强方面除了CTC方案中涉及的增强思路外, 尝试了cutmix, cutout, mixup, 最终发现只有mixup, cutout有用, 其它数据增强在一定程度下有用,但是会破坏数据, 会有瓶颈出现。

**训练结果:**
这个阶段的模型仅仅给数据增强方法筛选出来了, 最终筛选出cutout, mixup两种数据增强方法效果最好, 没有调参, 纯训练集B5模型单模单折平均线下0.974,五折交叉0.9829,如果加伪标签及调参可以继续提升, 第三种方案有做。

####2.4.MultiLabelSoftMargin loss 方案

**建模思路:**

由于 CrossEntropy loss 方案训练Loss与测试loss相差较大,比较难以观察调参, 故调整方案采用MultiLabelSoftMargin loss 方案, 该方案训练时有显示训练及验证集Loss,acc, 方便调参。

**训练结果:**

- 伪标签阈值则是根据模型输出结果softmax之后4个字符置信度最低的作为该样本置信度, 可以很好的筛选出好样本,也线上验证过筛选出的样本准确率在0.995+。
- 最终采用0.5概率mixup,0.5概率cutout,B3模型, 单模单折可以有0.978, 后期没来得及线上验证融合效果,保守预估可以达到0.985+, 线下验证分数对cutout的遮挡比例及遮挡矩形个数这两个参数比较敏感,不断调整预估单模单折线下可以达到0.98+,不过该方案是比赛后几天搞得,参数并没有调整到最优,具体调整到的参数值就不说了,有兴趣的同学自己调调参验证一下会有很好的经验收获。

##3.总结

参加这个比赛尽管换数据让大家不太开心, 但是不断调整方案及群里大家交流思路还是学到了非常多实际有用的知识的。

##4.数据集
链接：https://pan.baidu.com/s/1hiGOVeV4yO5gy2SVCx_z-g 
提取码：m947

##5.预训练模型文件

EfficientNet:链接：https://pan.baidu.com/s/1r-Lv7GSQQY_xsyf2310YJg 
提取码：ji5p

RepVGG:链接：https://pan.baidu.com/s/1FdpfxgpWxHHaKBKkpwSw5A 
提取码：97gx

##6.参考文献

链接：https://pan.baidu.com/s/14GnzL51wz-PY1yliy4mR9w 
提取码：jqwc

