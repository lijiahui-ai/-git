import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split        # 将数据分为训练集和测试集
from sklearn.preprocessing import StandardScaler                # 数据标准化
from sklearn.datasets import fetch_california_housing       # 加载加利福尼亚房价数据集

# 加载加利福尼亚房价数据集
california = fetch_california_housing()
X, y = california.data, california.target

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建模型
model = keras.Sequential()
model.add(layers.Input(shape=(X_train.shape[1],)))  # 输入层
model.add(layers.Dense(64, activation='relu'))  # 隐藏层
model.add(layers.Dense(32, activation='relu'))  # 隐藏层
model.add(layers.Dense(1))  # 输出层（回归任务）

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# 评估模型
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {test_loss:.4f}')

# 进行预测
predictions = model.predict(X_test)
print(predictions[:5])  # 输出前5个预测值
