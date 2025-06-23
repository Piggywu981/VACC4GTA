import cv2
import numpy as np
import pandas as pd
import mss
import mss.tools
import pyautogui
import keyboard
import time
import os
from datetime import datetime

# 配置参数
MINIMAP_REGION = {'top': 800, 'left': 20, 'width': 320, 'height': 320}  # 小地图区域坐标(需根据游戏窗口调整)
CAPTURE_INTERVAL = 0.1  # 采集间隔(秒)
DATA_SAVE_DIR = '../data/raw'
CSV_PATH = os.path.join(DATA_SAVE_DIR, 'training_data.csv')

# 确保数据目录存在
os.makedirs(DATA_SAVE_DIR, exist_ok=True)

def get_key_state():
    """获取当前按键状态，返回[left, forward, right]"""
    left = 1 if keyboard.is_pressed('a') else 0
    forward = 1 if keyboard.is_pressed('w') else 0
    right = 1 if keyboard.is_pressed('d') else 0
    
    # 确保只有一个方向被激活
    if sum([left, forward, right]) > 1:
        # 优先直行，然后左右
        if forward:
            left = 0
            right = 0
        elif left and right:
            left = 0
            right = 0
    
    return [left, forward, right]

def capture_minimap(sct):
    """捕获小地图图像并预处理"""
    screenshot = sct.grab(MINIMAP_REGION)
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)  # 转换颜色通道
    img = cv2.resize(img, (160, 120))  # 调整为模型输入尺寸
    return img

def main():
    print("数据采集开始 - 按ESC键停止")
    print(f"小地图区域: {MINIMAP_REGION}")
    print(f"数据保存至: {DATA_SAVE_DIR}")

    # 初始化CSV文件
    if not os.path.exists(CSV_PATH):
        pd.DataFrame(columns=['image_path', 'left', 'forward', 'right']).to_csv(CSV_PATH, index=False)

    with mss.mss() as sct:
        count = 0
        try:
            while True:
                # 检查退出条件
                if keyboard.is_pressed('esc'):
                    print("数据采集停止")
                    break

                # 获取按键状态
                key_state = get_key_state()
                if sum(key_state) == 0:
                    time.sleep(CAPTURE_INTERVAL)
                    continue

                # 捕获小地图
                img = capture_minimap(sct)

                # 保存图像
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                img_path = os.path.join(DATA_SAVE_DIR, f'frame_{timestamp}.jpg')
                cv2.imwrite(img_path, img)

                # 保存数据到CSV
                new_row = pd.DataFrame([{
                    'image_path': img_path,
                    'left': key_state[0],
                    'forward': key_state[1],
                    'right': key_state[2]
                }])
                new_row.to_csv(CSV_PATH, mode='a', header=False, index=False)

                count += 1
                if count % 100 == 0:
                    print(f"已采集 {count} 帧数据")

                time.sleep(CAPTURE_INTERVAL)

        except Exception as e:
            print(f"采集过程出错: {e}")

if __name__ == '__main__':
    main()