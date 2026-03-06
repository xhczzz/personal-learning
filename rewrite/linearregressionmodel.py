import torch
import numpy as np  
import matplotlib.pyplot as plt
import torch.nn as nn

torch.manual_seed(1)
X = torch.randn(50, 1) # <--- 改成 1 维特征
true_w = torch.tensor([2.0]) # <--- 对应的真实权重也只需要 1 个
true_b = 4.0
Y = X @ true_w + true_b + torch.randn(50) * 0.1

class linearregressionmodel(nn.Module):
    def __init__(self):
        super(linearregressionmodel, self).__init__()
        self.linear = nn.Linear(1, 1) # <--- 这里输入特征数改成 1
        
    def forward(self, x):
        return self.linear(x)

model = linearregressionmodel()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    predictions = model(X)
    loss = criterion(predictions.squeeze(), Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], loss: {loss.item():.4f}')


print(f'Predicted weight: {model.linear.weight.data.numpy()}')
print(f'Predicted bias: {model.linear.bias.data.numpy()}')

with torch.no_grad(): 
    predictions = model(X)

# 绘制散点图和拟合的直线
plt.scatter(X.numpy(), Y.numpy(), color='blue', label='True values')
plt.plot(X.numpy(), predictions.numpy(), color='red', label='Predictions') # 注意这里用 plt.plot 画线
plt.legend()
plt.show()