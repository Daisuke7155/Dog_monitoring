import torch
import cv2
import numpy as np
import torchvision.transforms as T
import os
import time
import pandas as pd
import signal

# YOLOv5リポジトリの正しいパスを指定
yolov5_path = os.path.join(os.path.dirname(__file__), 'yolov5')

# YOLOv5モデルのロード
model = torch.hub.load(yolov5_path, 'custom', path=os.path.join(yolov5_path, 'yolov5s.pt'), source='local')  # ローカルリポジトリを指定

# 行動分類モデルの定義
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes=6):  # クラス数を6に変更
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(64 * 56 * 56, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 行動分類モデルの読み込み
behavior_model = SimpleCNN(num_classes=6)
model_path = os.path.join(os.path.dirname(__file__), 'models/pet_behavior_model.pth')
behavior_model.load_state_dict(torch.load(model_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
behavior_model.to(device)
behavior_model.eval()

# クラスラベルの定義
classes = model.names
behavior_classes = ['barking', 'sleeping', 'awake', 'drinking', 'defecating', 'urinating']

# 画像の前処理
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor()
])

# 行動の時間計測用変数
current_action = None
start_time = None
action_records = []

# CSVに記録する関数
def save_to_csv():
    global action_records, current_action, start_time
    if current_action is not None and start_time is not None:
        end_time = time.time()
        duration = end_time - start_time
        action_records.append({
            'Action': current_action,
            'Start Time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
            'End Time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)),
            'Duration (s)': duration
        })
        print(f"{current_action} duration: {duration:.2f} seconds")
    
    # データフレームの作成
    df = pd.DataFrame(action_records)
    
    # CSVファイルに保存
    csv_path = os.path.join(os.path.dirname(__file__), 'action_durations.csv')
    df.to_csv(csv_path, index=False)
    print(f"Action durations saved to {csv_path}")

# シグナルハンドラの設定
def signal_handler(sig, frame):
    print('Interrupted! Saving data...')
    save_to_csv()
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# カメラの設定
cap = cv2.VideoCapture(0)  # カメラデバイスのインデックス

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv5モデルに画像を渡して予測を取得
        results = model(frame)

        # 結果の取得と行動分類
        for *box, conf, cls in results.xyxy[0]:
            label = int(cls)
            if classes[label] == 'dog':
                x1, y1, x2, y2 = map(int, box)
                roi = frame[y1:y2, x1:x2]
                roi_tensor = transform(roi).unsqueeze(0).to(device)

                # 行動分類
                with torch.no_grad():
                    behavior_outputs = behavior_model(roi_tensor)
                    softmax = torch.nn.Softmax(dim=1)
                    probs = softmax(behavior_outputs)
                    max_prob, predicted = torch.max(probs, 1)
                    
                    if max_prob.item() >= 0.5:
                        action = behavior_classes[predicted.item()]

                        # 行動の時間計測
                        end_time = time.time()
                        if current_action is not None and start_time is not None:
                            duration = end_time - start_time
                            action_records.append({
                                'Action': current_action,
                                'Start Time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
                                'End Time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)),
                                'Duration (s)': duration
                            })
                            print(f"{current_action} duration: {duration:.2f} seconds")

                        current_action = action
                        start_time = end_time

                        # バウンダリボックスと行動ラベルの描画
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'Dog: {conf:.2f}, Action: {action}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # 結果の表示
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    save_to_csv()
    cap.release()
    cv2.destroyAllWindows()
