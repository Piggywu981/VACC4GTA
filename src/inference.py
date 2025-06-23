import cv2
import numpy as np
import torch
from model import AlexNet
import mss
import pyautogui
import keyboard
import time
import os

# 配置参数
MODEL_PATH = '../models/selfdrive_model.pth'
MINIMAP_REGION = {'top': 800, 'left': 1600, 'width': 320, 'height': 320}  # 需与采集脚本保持一致
IMAGE_SIZE = (160, 120)
CONTROL_DELAY = 0.1  # 控制指令发送间隔(秒)

# 确保中文显示正常
pyautogui.FAILSAFE = False

class SelfDriveInference:
    def __init__(self):
        # 加载训练好的模型
        self.model = self.load_model()
        # 初始化屏幕捕获
        self.sct = mss.mss()
        # 控制状态
        self.running = False

    def load_model(self):
        """加载训练好的PyTorch模型"""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型实例
        input_shape = (*IMAGE_SIZE, 3)  # (160, 120, 3)
        model = AlexNet(input_shape)
        
        # 加载模型权重
        model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        model.to(self.device)
        model.eval()  # 设置为评估模式
        
        return model

    def preprocess_image(self, img):
        """预处理图像以匹配PyTorch模型输入"""
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        img = img / 255.0
        
        # 转换为PyTorch格式 (H, W, C) -> (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        
        # 转换为Tensor并添加批次维度
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        return img_tensor.to(self.device)

    def get_control_command(self, prediction):
        """根据模型预测获取控制指令"""
        predicted_class = np.argmax(prediction)
        # 0: 左转, 1: 直行, 2: 右转
        return predicted_class

    def execute_control(self, command):
        """执行控制指令"""
        # 先释放所有按键
        pyautogui.keyUp('a')
        pyautogui.keyUp('w')
        pyautogui.keyUp('d')

        # 根据指令按下对应按键
        if command == 0:  # 左转
            pyautogui.keyDown('a')
            pyautogui.keyDown('w')  # 左转时保持前进
            print("控制指令: 左转")
        elif command == 1:  # 直行
            pyautogui.keyDown('w')
            print("控制指令: 直行")
        elif command == 2:  # 右转
            pyautogui.keyDown('d')
            pyautogui.keyDown('w')  # 右转时保持前进
            print("控制指令: 右转")

    def run(self):
        """运行自动驾驶推理"""
        print("自动驾驶推理开始 - 按ESC键停止")
        self.running = True

        try:
            while self.running:
                # 检查退出条件
                if keyboard.is_pressed('esc'):
                    self.running = False
                    break

                # 捕获小地图
                screenshot = self.sct.grab(MINIMAP_REGION)
                img = np.array(screenshot)

                # 预处理图像
                processed_img = self.preprocess_image(img)

                # 模型预测
                with torch.no_grad():
                    output = self.model(processed_img)
                    probabilities = torch.softmax(output, dim=1)
                    prediction = probabilities.cpu().numpy()

                # 获取并执行控制指令
                command = self.get_control_command(prediction)
                self.execute_control(command)

                # 控制频率
                time.sleep(CONTROL_DELAY)

        except Exception as e:
            print(f"推理过程出错: {e}")
        finally:
            # 释放所有按键
            pyautogui.keyUp('a')
            pyautogui.keyUp('w')
            pyautogui.keyUp('d')
            print("自动驾驶推理停止")

if __name__ == '__main__':
    try:
        inference = SelfDriveInference()
        inference.run()
    except Exception as e:
        print(f"启动失败: {e}")