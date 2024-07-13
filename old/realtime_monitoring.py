import os
import torch
import cv2
import torchvision.transforms as transforms
import torchvision
from dog_model import SimpleCNN

# モデルのインスタンスを作成
behavior_model = SimpleCNN(num_classes=4)

# モデルの読み込み
model_load_path = os.path.join(os.path.dirname(__file__), './models/pet_behavior_model.pth')
behavior_model.load_state_dict(torch.load(model_load_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
behavior_model.to(device)
behavior_model.eval()

# クラスラベルの定義
classes = ['drinking', 'sleeping', 'barking', 'defecating']

# 物体検出モデルの準備
detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.COCO_V1')
detection_model.to(device)
detection_model.eval()

# 画像の前処理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# カメラの設定
cap = cv2.VideoCapture(0)  # カメラデバイスのインデックス

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 画像の前処理
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    image_tensor = transform(image).to(device)

    # 物体検出
    with torch.no_grad():
        detection_outputs = detection_model([image_tensor])[0]

    for box, score, label in zip(detection_outputs['boxes'], detection_outputs['scores'], detection_outputs['labels']):
        if score >= 0.5 and label == 1:  # スコアが0.5以上かつラベルが犬の場合
            x1, y1, x2, y2 = box.int().tolist()
            roi = original_image[y1:y2, x1:x2]
            roi_tensor = transform(roi).unsqueeze(0).to(device)

            # 行動分類
            with torch.no_grad(): 
                behavior_outputs = behavior_model(roi_tensor)
                _, predicted = torch.max(behavior_outputs, 1)
                action = classes[predicted.item()]

            # バウンダリボックスと行動ラベルの描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, action, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # 結果の表示
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
