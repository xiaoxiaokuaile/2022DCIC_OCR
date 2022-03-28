# Learner: 王振强
# Learn Time: 2022/2/16 14:40
# Learner: 王振强
# Learn Time: 2022/2/9 13:16
import timm
import torch

"""
    timm视觉库的介绍
    安装: pip install timm
"""


# 查看已经可以直接创建,有预训练参数的模型列表
all_pretrained_models_available = timm.list_models(pretrained=True)
# 通过通配符搜索模型体系结构
all_densenet_models = timm.list_models('*EfficientNet*',pretrained=True)
# densenet: ['densenet121', 'densenet161', 'densenet169', 'densenet201', 'densenetblur121d', 'tv_densenet121']
# resnet18: ['gluon_resnet18_v1b', 'legacy_seresnet18', 'resnet18', 'resnet18d', 'skresnet18', 'ssl_resnet18', 'swsl_resnet18']
# resnet34: ['gluon_resnet34_v1b', 'legacy_seresnet34', 'resnet34', 'resnet34d', 'skresnet34', 'tv_resnet34']
# resnet50: ['cspresnet50', 'ecaresnet50d', 'ecaresnet50d_pruned', 'ecaresnet50t', 'gluon_resnet50_v1b',
#            'gluon_resnet50_v1c', 'gluon_resnet50_v1d', 'gluon_resnet50_v1s', 'legacy_seresnet50', 'nf_resnet50',
#            'resnet50', 'resnet50d', 'seresnet50', 'ssl_resnet50', 'swsl_resnet50', 'tv_resnet50', 'wide_resnet50_2']
# resnet101:['ecaresnet101d', 'ecaresnet101d_pruned', 'gluon_resnet101_v1b', 'gluon_resnet101_v1c', 'gluon_resnet101_v1d',
#            'gluon_resnet101_v1s', 'legacy_seresnet101', 'resnet101d', 'tv_resnet101', 'wide_resnet101_2']
print(all_densenet_models)

# ---------------------- 创建模型 -----------------------
# 创建输入图像 (bz,3,H,W)
x = torch.randn(2, 3, 128, 320)

# 创建 resnet-34
model1 = timm.create_model('resnet34',pretrained=True)
# print(model1(x).shape) # torch.Size([1, 1000])
# 创建 resnet-101e
model2 = timm.create_model('resnest101e',pretrained=True)
# 创建具有自定义类数的模型,只需传入 num_classes
model3 = timm.create_model('resnet34', num_classes=10)
# print(model3(x).shape) # torch.Size([1, 10])

# -------------------- 查看特征提取器 ----------------------
all_feature_extractor = timm.create_model('resnet18', features_only=True)
all_features = all_feature_extractor(x)
print('All {} Features: '.format(len(all_features)))
for i in range(len(all_features)):
    print('feature {} shape: {}'.format(i, all_features[i].shape))

# 提取中间层特征
out_indices = [2, 3, 4]
selected_feature_extractor = timm.create_model('resnet50', features_only=True, out_indices=out_indices)
selected_features = selected_feature_extractor(x)
# print('Selected Features: ')
# for i in range(len(out_indices)):
#     print('feature {} shape: {}'.format(out_indices[i], selected_features[i].shape))

# --------------------- 查看特征提取器类型 ---------------------
print('type:', type(all_feature_extractor))
print('len: ', len(all_feature_extractor))
# len:  8
# net: conv1, bn1, act1, maxpool, layer1, layer2, layer3, layer4
# resnet18:                64      64      128     256     512
# resnet34:                64      64      128     256     512
# resnet50:                64      256     512     1024    2048
# resnet101:               64      256     512     1024    2048
# resnet101e              128      256     512     1024    2048
# densenet121:             64      256     512     1024    1024
for item in all_feature_extractor:
    print(item)






















