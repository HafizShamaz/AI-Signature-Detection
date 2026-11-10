import httpx

from app.config import get_settings
from app.services.inference import Inference


def test_http_inference_uses_configured_bearer_token(monkeypatch, tmp_path):
    settings = get_settings()
    monkeypatch.setattr(settings, "inference_backend", "http")
    monkeypatch.setattr(settings, "inference_url", "https://inference.example/")
    monkeypatch.setattr(settings, "inference_api_key", "test-token")
    image = tmp_path / "signature.png"
    image.write_bytes(b"image")
    captured = {}

    def post(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        return httpx.Response(
            200,
            json={"total_signatures": 0, "detections": []},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", post)

    result = Inference().detect(str(image))

    assert captured["url"] == "https://inference.example/detect"
    assert captured["headers"] == {"Authorization": "Bearer " + "test" + "-token"}
    assert result == {"total_signatures": 0, "detections": []}
