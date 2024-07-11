import torch
import cv2
import torchvision.transforms as transforms

# モデルのインスタンスを作成
model = SimpleCNN(num_classes=4)

# 保存したパラメータを読み込む
model.load_state_dict(torch.load('pet_behavior_model.pth'))
model.to(device)
model.eval()

print('Model loaded successfully.')

# カメラの設定
cap = cv2.VideoCapture(0)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        action = dataset.classes[predicted.item()]

    cv2.putText(frame, action, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
