charNumber = 4
# 图片宽高
ImageWidth = 320  # 320
ImageHeight = 128 # 128

# 图像分类的类别
num_classes = 248

learningRate = 1e-3  # 3e-4 #3.5e-4  # 1e-2
totalEpoch = 60
batchSize = 32   # 32

# MixUp 超参数
alpha = 0.5
mixup_prob = 0.5


# 数据集路径
img_path = "./data/training_dataset"
# 测试集图片地址
testPath = "./data/test"