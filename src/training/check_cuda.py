import torch
import torch.nn as nn
import time
import os

print("=== CPU模式训练 ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"设备: CPU")
print(f"线程数: {torch.get_num_threads()}")

# 设置CPU优化
torch.set_num_threads(min(8, os.cpu_count()))


# 简单的模型
class SimpleModel(nn.Module):
    def __init__(self, input_size=100, hidden_size=50, output_size=1):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 创建模型和数据
device = torch.device('cpu')
model = SimpleModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 生成模拟数据
batch_size = 64
x_train = torch.randn(1000, 100)
y_train = torch.randn(1000, 1)

print(f"\n开始训练...")
print(f"数据形状: {x_train.shape}")

# 训练循环
model.train()
start_time = time.time()

for epoch in range(10):  # 只训练10个epoch测试
    epoch_loss = 0.0

    for i in range(0, len(x_train), batch_size):
        batch_x = x_train[i:i + batch_size]
        batch_y = y_train[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if epoch % 2 == 0:
        avg_loss = epoch_loss / (len(x_train) // batch_size)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

end_time = time.time()
print(f"\n训练完成! 耗时: {end_time - start_time:.2f}秒")

# 测试推理
model.eval()
with torch.no_grad():
    test_input = torch.randn(10, 100)
    test_output = model(test_input)
    print(f"\n推理测试:")
    print(f"输入: {test_input.shape}")
    print(f"输出: {test_output.shape}")
    print("✅ CPU模式运行成功!")