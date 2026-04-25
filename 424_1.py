# 0. 调用要使用的包
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import random

np.random.seed(5)

# 1. 生成数据集
x_train = np.random.random((1000, 12))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 12))
y_test = np.random.randint(2, size=(100, 1))

# 2. 搭建模型
model = Sequential()
model.add(Dense(1, input_dim=12, activation='sigmoid'))

# 3. 设置模型训练过程
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#optimizer：优化器； loss：计算损失，这里用的是交叉熵损失； metrics: 列表，包含评估模型在训练和测试时的性能的指标

# 4. 训练模型
hist = model.fit(x_train, y_train, epochs=1000, batch_size=64)

# 5. 查看训练过程
#%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.set_ylim([0.0, 1.0])
acc_ax.set_ylim([0.0, 1.0])
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()

# 6. 评价模型
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('loss_and_metrics : ' + str(loss_and_metrics))
