import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# データセットクラスの定義
class PetBehaviorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['drinking', 'barking','sleeping', 'awake', 'defecating', 'urinating']
        self.data = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.data.append((img_path, class_name))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        label = self.classes.index(class_name)
        return image, label

# 画像の変換
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# データセットの作成
dataset = PetBehaviorDataset(root_dir='dataset', transform=transform)

# データセットをトレーニングとテストに分割
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# データローダーの作成
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# CNNモデルの定義
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):  # クラス数を6に変更
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# モデルのトレーニング
model = SimpleCNN(num_classes=6)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):  # エポック数を指定
    model.train()  # モデルをトレーニングモードに設定
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:  # 10バッチごとにログを出力
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0

print('Finished Training')

# モデルのテスト
model.eval()  # モデルを評価モードに設定
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {total} test images: {100 * correct / total:.2f}%')

# 各クラスごとの精度を計算するための準備
class_correct = [0] * 6
class_total = [0] * 6

model.eval()  # モデルを評価モードに設定
all_labels = []
all_preds = []

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# 各クラスごとの精度を表示
for i in range(6):
    print(f'Accuracy of {dataset.classes[i]} : {100 * class_correct[i] / class_total[i]:.2f}%')

# 混同行列を表示
conf_matrix = confusion_matrix(all_labels, all_preds)
print('Confusion Matrix')
print(conf_matrix)

# クラスごとの詳細なレポートを表示
class_report = classification_report(all_labels, all_preds, target_names=dataset.classes)
print('Classification Report')
print(class_report)

# モデルの保存
torch.save(model.state_dict(), 'models/pet_behavior_model.pth')
print('Model saved successfully.')
