"""
FastAPI 모델 서빙 API
- 이미지 업로드 → 숫자 예측
- 헬스체크 엔드포인트
- 모델 메타데이터 조회
"""

import io
import os
import sys
from datetime import datetime

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

# 상위 디렉토리의 모델 import
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import get_model  # noqa: E402

# FastAPI 앱 초기화
app = FastAPI(
    title="MNIST 숫자 분류 API",
    description="손글씨 숫자(0-9) 이미지를 분류하는 MLOps 데모 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 전역 변수
model = None
device = None
transform = None
model_info = {}


def load_model():
    """모델 로드"""
    global model, device, transform, model_info

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(num_classes=10)

    model_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "best_model.pth"
    )

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model_info = {
            "loaded": True,
            "epoch": checkpoint.get("epoch", "N/A"),
            "accuracy": checkpoint.get("accuracy", "N/A"),
            "path": model_path,
        }
        print(f"✅ Model loaded from {model_path}")
    else:
        model_info = {
            "loaded": False,
            "message": "Using randomly initialized model",
        }
        print(f"⚠️  No saved model found at {model_path}")

    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


@app.on_event("startup")
async def startup_event():
    """앱 시작 시 모델 로드"""
    load_model()


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "MNIST 숫자 분류 API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "이미지 업로드하여 숫자 예측",
            "GET /health": "서버 상태 확인",
            "GET /model/info": "모델 정보 조회",
            "GET /docs": "API 문서 (Swagger UI)",
        },
    }


@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트 - Kubernetes/Docker 헬스체크용"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model is not None,
        "device": str(device),
    }


@app.get("/model/info")
async def get_model_info():
    """모델 메타데이터 조회"""
    return {
        "model_type": "SimpleCNN",
        "num_classes": 10,
        "input_shape": [1, 28, 28],
        "device": str(device),
        **model_info,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    이미지 업로드 → 숫자 예측

    - **file**: 손글씨 숫자 이미지 (PNG, JPG 등)

    Returns:
        - predicted_digit: 예측된 숫자 (0-9)
        - confidence: 예측 확신도 (0-1)
        - all_probabilities: 각 숫자별 확률
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        all_probs = probabilities.squeeze().cpu().numpy().tolist()

        return JSONResponse(
            content={
                "predicted_digit": int(predicted.item()),
                "confidence": round(float(confidence.item()), 4),
                "all_probabilities": {
                    str(i): round(prob, 4) for i, prob in enumerate(all_probs)
                },
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    여러 이미지 배치 예측

    - **files**: 여러 손글씨 숫자 이미지
    """
    results = []

    for file in files:
        if not file.content_type.startswith("image/"):
            results.append(
                {"filename": file.filename, "error": "이미지 파일이 아닙니다."}
            )
            continue

        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            results.append(
                {
                    "filename": file.filename,
                    "predicted_digit": int(predicted.item()),
                    "confidence": round(float(confidence.item()), 4),
                }
            )

        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return {"results": results, "total": len(results)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
