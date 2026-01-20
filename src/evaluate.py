"""
모델 평가 스크립트
- 학습된 모델 로드
- 테스트 데이터셋으로 평가
- 결과를 JSON으로 저장
"""

import argparse
import json
import os

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import get_model


def load_model(model_path: str, device: torch.device):
    """학습된 모델 로드"""
    model = get_model(num_classes=10)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ Model loaded from {model_path}")
        print(f"   Trained accuracy: {checkpoint.get('accuracy', 'N/A')}%")
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")

    model.to(device)
    model.eval()
    return model


def get_test_loader(batch_size: int = 64, data_dir: str = "../data"):
    """테스트 데이터 로더"""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return test_loader


def evaluate_model(model, test_loader, device):
    """모델 평가"""
    model.eval()
    all_predictions = []
    all_targets = []
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100.0 * correct / total

    # Classification Report
    report = classification_report(
        all_targets,
        all_predictions,
        target_names=[str(i) for i in range(10)],
        output_dict=True,
    )

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_predictions)

    return {
        "accuracy": round(accuracy, 2),
        "total_samples": total,
        "correct_predictions": correct,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def main(args):
    """메인 함수"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # 모델 로드
    model = load_model(args.model_path, device)

    # 테스트 데이터 로더
    test_loader = get_test_loader(data_dir=args.data_dir)
    print(f"📊 Test samples: {len(test_loader.dataset)}")

    # 평가 실행
    print("\n🔍 Evaluating model...")
    results = evaluate_model(model, test_loader, device)

    # 결과 출력
    print("\n📈 Evaluation Results:")
    print(f"   Accuracy: {results['accuracy']}%")
    print(f"   Correct: {results['correct_predictions']}/{results['total_samples']}")

    # 결과 저장
    output_path = os.path.join(args.output_dir, "evaluation_results.json")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")

    # 품질 게이트 체크
    if results["accuracy"] >= args.min_accuracy:
        print(f"✅ Model PASSED quality gate (>= {args.min_accuracy}%)")
        return 0
    else:
        print(f"❌ Model FAILED quality gate (< {args.min_accuracy}%)")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모델 평가")

    parser.add_argument(
        "--model-path",
        type=str,
        default="../models/best_model.pth",
        help="모델 파일 경로",
    )
    parser.add_argument("--data-dir", type=str, default="../data", help="데이터 경로")
    parser.add_argument(
        "--output-dir", type=str, default="../models", help="결과 저장 경로"
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=95.0,
        help="최소 정확도 임계값 (%)",
    )

    args = parser.parse_args()

    exit_code = main(args)
    exit(exit_code)
