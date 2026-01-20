"""
CNN 모델 정의
MNIST 손글씨 숫자 분류 (0-9)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    간단한 CNN 모델
    - 2개의 Convolutional Layer
    - 2개의 Fully Connected Layer
    - Dropout으로 과적합 방지
    """

    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Pooling & Dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv Block 1: 28x28 -> 14x14
        x = self.pool(F.relu(self.conv1(x)))

        # Conv Block 2: 14x14 -> 7x7
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)

        # Flatten
        x = x.view(-1, 64 * 7 * 7)

        # FC Layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """추론용 메서드 - softmax 확률 반환"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities


def get_model(num_classes: int = 10) -> SimpleCNN:
    """모델 인스턴스 생성 팩토리 함수"""
    return SimpleCNN(num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    """모델 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
