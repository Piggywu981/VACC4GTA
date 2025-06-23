import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# 配置参数
IMAGE_SIZE = (160, 120)
INPUT_SHAPE = (*IMAGE_SIZE, 3)
BATCH_SIZE = 32
EPOCHS = 50
DATA_PATH = '../data/processed/balanced_training_data.csv'
MODEL_SAVE_PATH = '../models/selfdrive_model.pth'

# 确保模型保存目录存在
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

class DrivingDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = self.df.iloc[idx]['image_path']
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"图像文件不存在: {img_path}")
        
        # 转换为RGB并归一化
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        
        # 转换为PyTorch格式 (H, W, C) -> (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32)
        
        # 获取标签
        label = self.df.iloc[idx][['left', 'forward', 'right']].values
        label = torch.tensor(label, dtype=torch.long)
        label = torch.argmax(label)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def load_data(csv_path, batch_size=32):
    """加载训练数据并创建DataLoader"""
    # 定义数据变换
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5)
    ])
    
    # 创建数据集
    dataset = DrivingDataset(csv_path, transform=transform)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

class AlexNet(nn.Module):
    def __init__(self, input_shape, num_classes=3):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 第二层卷积
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 第三、四、五层卷积
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # 计算全连接层输入特征数
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1])
            features = self.features(dummy_input)
            self.num_flat_features = features.view(1, -1).size(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.num_flat_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.num_flat_features)
        x = self.classifier(x)
        return x


def build_alexnet_model(input_shape, num_classes=3):
    """构建基于AlexNet的模型"""
    return AlexNet(input_shape, num_classes)

def train_model():
    """训练模型并保存"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    train_loader, val_loader = load_data(DATA_PATH, batch_size=BATCH_SIZE)
    print(f"数据加载完成 - 训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")
    
    # 构建模型
    model = build_alexnet_model(INPUT_SHAPE).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练参数
    best_val_acc = 0.0
    patience = 5
    early_stop_counter = 0
    
    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 训练批次
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # 计算训练集指标
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100 * train_correct / train_total
        
        # 验证批次
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # 计算验证集指标
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total
        
        # 打印 epoch 结果
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')
        
        # 模型保存和早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'模型保存至 {MODEL_SAVE_PATH} (验证准确率: {best_val_acc:.2f}%)')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f'早停计数: {early_stop_counter}/{patience}')
            if early_stop_counter >= patience:
                print('早停条件触发，停止训练')
                break
    
    print(f"模型训练完成，最佳验证准确率: {best_val_acc:.2f}%")
    return model

def main():
    try:
        train_model()
    except Exception as e:
        print(f"模型训练出错: {e}")

if __name__ == '__main__':
    main()