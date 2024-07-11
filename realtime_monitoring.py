import torch
import cv2
import torchvision.transforms as transforms
import torchvision
from dog_model import SimpleCNN  # 既存のモデル定義をインポート

# 行動分類モデルの読み込み
behavior_model = SimpleCNN(num_classes=4)
behavior_model.load_state_dict(torch.load('../models/dog_behavior_model.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
behavior_model.to(device)
behavior_model.eval()

# 物体検出モデルの読み込み
detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
detection_model.to(device)
detection_model.eval()

classes = ['drinking', 'sleeping', 'barking', 'defecating']
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)  # カメラデバイスのインデックス

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        detection_outputs = detection_model([image_tensor])[0]

    for box, score, label in zip(detection_outputs['boxes'], detection_outputs['scores'], detection_outputs['labels']):
        if score >= 0.5 and label == 1:  # スコアが0.5以上かつラベルが犬の場合
            x1, y1, x2, y2 = box.int().tolist()
            roi = frame[y1:y2, x1:x2]
            roi_tensor = transform(roi).unsqueeze(0).to(device)

            with torch.no_grad():
                behavior_outputs = behavior_model(roi_tensor)
                _, predicted = torch.max(behavior_outputs, 1)
                action = classes[predicted.item()]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, action, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
