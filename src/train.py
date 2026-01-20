"""
모델 학습 스크립트
- MNIST 데이터셋 로드
- 모델 학습
- MLflow로 실험 추적
"""

import argparse
import os
from datetime import datetime

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import count_parameters, get_model


def get_data_loaders(batch_size: int = 64, data_dir: str = "../data"):
    """MNIST 데이터 로더 생성"""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """한 에폭 학습"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """모델 평가"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def train(args):
    """메인 학습 함수"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # MLflow 설정
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # 하이퍼파라미터 로깅
        mlflow.log_params(
            {
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "optimizer": "Adam",
                "device": str(device),
            }
        )

        # 데이터 로더
        train_loader, test_loader = get_data_loaders(
            batch_size=args.batch_size, data_dir=args.data_dir
        )
        print(f"📊 Train samples: {len(train_loader.dataset)}")
        print(f"📊 Test samples: {len(test_loader.dataset)}")

        # 모델, 손실함수, 옵티마이저
        model = get_model(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # 모델 정보 로깅
        mlflow.log_param("model_parameters", count_parameters(model))

        # 학습 루프
        best_accuracy = 0.0

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )

            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            # 메트릭 로깅
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                },
                step=epoch,
            )

            print(f"Epoch {epoch}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

            # Best 모델 저장
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                model_path = os.path.join(args.model_dir, "best_model.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "accuracy": test_acc,
                    },
                    model_path,
                )
                print(f"  ✅ Best model saved! (Accuracy: {test_acc:.2f}%)")

        # 최종 모델 MLflow에 저장
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_metric("best_accuracy", best_accuracy)

        print(f"\n🎉 Training completed! Best accuracy: {best_accuracy:.2f}%")
        print(f"📁 Model saved to: {args.model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST 분류 모델 학습")

    parser.add_argument("--batch-size", type=int, default=64, help="배치 크기")
    parser.add_argument("--epochs", type=int, default=5, help="에폭 수")
    parser.add_argument("--lr", type=float, default=0.001, help="학습률")

    parser.add_argument("--data-dir", type=str, default="../data", help="데이터 경로")
    parser.add_argument(
        "--model-dir", type=str, default="../models", help="모델 저장 경로"
    )

    parser.add_argument(
        "--mlflow-uri", type=str, default="mlruns", help="MLflow 추적 URI"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="mnist-classification",
        help="실험 이름",
    )

    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    train(args)
