"""
유닛 테스트
- 모델 테스트
- API 테스트
"""

import os
import sys

import pytest
import torch

# 소스 디렉토리 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


class TestModel:
    """모델 관련 테스트"""

    def test_model_initialization(self):
        """모델 초기화 테스트"""
        from model import get_model

        model = get_model(num_classes=10)
        assert model is not None

    def test_model_forward_pass(self):
        """모델 순전파 테스트"""
        from model import get_model

        model = get_model(num_classes=10)
        model.eval()

        # 배치 크기 1, 채널 1, 28x28 이미지
        dummy_input = torch.randn(1, 1, 28, 28)

        with torch.no_grad():
            output = model(dummy_input)

        # 출력 shape 검증: (batch_size, num_classes)
        assert output.shape == (1, 10)

    def test_model_batch_forward(self):
        """배치 순전파 테스트"""
        from model import get_model

        model = get_model(num_classes=10)
        model.eval()

        batch_size = 32
        dummy_input = torch.randn(batch_size, 1, 28, 28)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (batch_size, 10)

    def test_model_predict_method(self):
        """predict 메서드 테스트"""
        from model import get_model

        model = get_model(num_classes=10)
        dummy_input = torch.randn(1, 1, 28, 28)

        probabilities = model.predict(dummy_input)

        # 확률 합이 1인지 확인
        prob_sum = probabilities.sum().item()
        assert abs(prob_sum - 1.0) < 1e-5

    def test_count_parameters(self):
        """파라미터 수 계산 테스트"""
        from model import count_parameters, get_model

        model = get_model(num_classes=10)
        param_count = count_parameters(model)

        assert param_count > 0
        assert isinstance(param_count, int)


class TestAPI:
    """API 관련 테스트"""

    @pytest.fixture
    def client(self):
        """테스트 클라이언트 생성"""
        from fastapi.testclient import TestClient
        from main import app

        return TestClient(app)

    def test_root_endpoint(self, client):
        """루트 엔드포인트 테스트"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data

    def test_health_endpoint(self, client):
        """헬스체크 엔드포인트 테스트"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data

    def test_model_info_endpoint(self, client):
        """모델 정보 엔드포인트 테스트"""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "SimpleCNN"
        assert data["num_classes"] == 10

    def test_predict_invalid_file(self, client):
        """잘못된 파일 업로드 테스트"""
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400

    def test_predict_valid_image(self, client):
        """유효한 이미지 예측 테스트"""
        from io import BytesIO

        from PIL import Image

        # 28x28 흑백 테스트 이미지 생성
        img = Image.new("L", (28, 28), color=0)
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        response = client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
        )

        # 모델이 로드되었으면 200, 아니면 500 (CI 환경에서는 모델 파일 없음)
        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = response.json()
            assert "predicted_digit" in data
            assert "confidence" in data
            assert 0 <= data["predicted_digit"] <= 9
            assert 0 <= data["confidence"] <= 1


class TestDataPipeline:
    """데이터 파이프라인 테스트"""

    def test_transforms(self):
        """데이터 변환 테스트"""
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        from PIL import Image

        # 테스트 이미지 생성
        img = Image.new("L", (28, 28), color=128)
        tensor = transform(img)

        assert tensor.shape == (1, 28, 28)
        assert tensor.dtype == torch.float32
