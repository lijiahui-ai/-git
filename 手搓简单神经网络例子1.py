#这是chatGPT写的

import numpy as np


# 激活函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)

#这个网络包括一个输入层，一个隐藏层和一个输出层，并使用反向传播算法来训练。
# 定义神经网络类
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化网络参数

        self.hidden_size = hidden_size
        self.output_size = output_size

        # 权重和偏置初始化
        self.input_size = input_size
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.random.randn(1, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.random.randn(1, self.output_size)

    def forward(self, X):
        # self.input_size = X.shape[1]
        # self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        # 前向传播
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y, learning_rate):
        # 反向传播 ,相关知识看“神经网络与深度学习(邱锡鹏)”4.4节
        output_error = y - self.final_output
        output_delta = output_error * sigmoid_derivative(self.final_output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # 更新权重和偏置       #好像有点不对
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        # 训练网络
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - self.final_output))
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        # 预测输出
        return self.forward(X)


# 输入数据和目标输出（例如逻辑与运算）
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [0],
              [0],
              [1]])

# 创建神经网络实例
input_size = 2  # 输入层节点数
hidden_size = 3  # 隐藏层节点数
output_size = 1  # 输出层节点数

nn = NeuralNetwork(input_size, hidden_size, output_size)

# 训练神经网络
nn.train(X, y, epochs=10000, learning_rate=0.1)

# 测试
print("Predictions:")
print(nn.predict(X))
