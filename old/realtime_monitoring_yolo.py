import torch
import cv2
import numpy as np
import torchvision.transforms as T
import os
import sys

# YOLOv5モデルのロード
model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local')  # ローカルリポジトリを指定

# 行動分類モデルの定義
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes=4):
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
behavior_model = SimpleCNN(num_classes=4)
model_path = os.path.join(os.path.dirname(__file__), 'models/pet_behavior_model.pth')
behavior_model.load_state_dict(torch.load(model_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
behavior_model.to(device)
behavior_model.eval()

# クラスラベルの定義
classes = model.names
behavior_classes = ['drinking', 'sleeping', 'barking', 'defecating']

# 画像の前処理
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor()
])

# カメラの設定
cap = cv2.VideoCapture(0)  # カメラデバイスのインデックス

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
                _, predicted = torch.max(behavior_outputs, 1)
                action = behavior_classes[predicted.item()]

            # バウンダリボックスと行動ラベルの描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Dog: {conf:.2f}, Action: {action}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # 結果の表示
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
