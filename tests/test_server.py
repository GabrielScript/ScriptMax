from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scriptmax.server import STATIC_DIR, create_app
from tests.conftest import build_services

LOCAL = {"base_url": "http://127.0.0.1:8000", "client": ("127.0.0.1", 50000)}
TOKEN = "token-super-secreto-123"


def wait_for_job(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["stage"] in {"done", "failed"}:
            return job
        time.sleep(0.05)
    raise AssertionError("job não terminou")


def upload(client: TestClient, **fields: str) -> object:
    data = {"subject": "Álgebra", "category": "academico", "folder": "Álgebra Linear/Prova 1"} | fields
    return client.post("/api/jobs", data=data, files={"file": ("aula.m4a", b"fake-audio", "audio/mp4")})


@pytest.fixture()
def local_client(tmp_path: Path, fake_pdf: None):
    with TestClient(create_app(build_services(tmp_path)), **LOCAL) as client:
        yield client


def test_full_flow_creates_report_in_folder(local_client: TestClient, tmp_path: Path) -> None:
    response = upload(local_client)
    assert response.status_code == 202, response.text
    job = wait_for_job(local_client, response.json()["id"])
    assert job["stage"] == "done", job

    reports = local_client.get("/api/reports").json()
    assert len(reports) == 1
    report = reports[0]
    assert report["folder"] == "Álgebra Linear/Prova 1"
    assert report["library_pdf"].startswith("Acadêmico e Conhecimento/Álgebra Linear/Prova 1/")
    assert (tmp_path / "biblioteca" / report["library_pdf"]).exists()
    assert not any((tmp_path / "uploads").iterdir())  # upload apagado após sucesso

    html = local_client.get(f"/api/reports/{report['id']}/files/html")
    assert html.status_code == 200
    assert "sandbox allow-scripts" in html.headers["content-security-policy"]
    assert 'class="tex2jax_ignore">R$</span>' in html.text


def test_move_updates_library_copy(local_client: TestClient, tmp_path: Path) -> None:
    job = wait_for_job(local_client, upload(local_client).json()["id"])
    report_id = job["report_id"]
    moved = local_client.patch(f"/api/reports/{report_id}", json={"category": "trabalho", "folder": "Cliente A"}).json()
    assert moved["library_pdf"].startswith("Trabalho/Cliente A/")
    assert (tmp_path / "biblioteca" / moved["library_pdf"]).exists()
    assert not (tmp_path / "biblioteca" / "Acadêmico e Conhecimento" / "Álgebra Linear").exists()

    assert local_client.delete(f"/api/reports/{report_id}").status_code == 204
    assert not (tmp_path / "biblioteca" / moved["library_pdf"]).exists()


def test_rejects_bad_input(local_client: TestClient) -> None:
    assert upload(local_client, subject="   ").status_code == 400
    assert upload(local_client, folder="a/b/c/d").status_code == 400
    assert upload(local_client, category="inexistente").status_code == 422
    bad_ext = local_client.post("/api/jobs", data={"subject": "x", "category": "trabalho"},
                                files={"file": ("virus.exe", b"MZ", "application/octet-stream")})
    assert bad_ext.status_code == 400


def test_rejects_oversized_upload_before_parsing(local_client: TestClient) -> None:
    big = b"0" * (2 * 1024 * 1024)
    response = local_client.post("/api/jobs", data={"subject": "x", "category": "trabalho"},
                                 files={"file": ("a.mp3", big, "audio/mpeg")})
    assert response.status_code == 413


def test_invalid_report_id_is_404(local_client: TestClient) -> None:
    assert local_client.get("/api/reports/..%2F..%2Fsecret/files/html").status_code == 404
    assert local_client.get(f"/api/reports/{'0' * 32}/files/html").status_code == 404


def test_cross_origin_post_is_blocked(local_client: TestClient) -> None:
    response = local_client.post(f"/api/reports/{'0' * 32}/regenerate", headers={"Origin": "https://evil.example"})
    assert response.status_code == 403


def test_security_headers_present(local_client: TestClient) -> None:
    response = local_client.get("/")
    assert response.status_code == 200
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["strict-transport-security"] == "max-age=31536000; includeSubDomains"
    assert "x-request-id" in response.headers


def test_forms_never_submit_natively_via_get() -> None:
    # Sem JS (ou antes do módulo carregar), um <form> sem method faria GET /?token=... e o
    # token iria parar no log de requisições do Cloud Run e no histórico do navegador.
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert '<form id="login-form" method="post"' in html
    assert '<form id="details-form" class="details" method="post"' in html


def test_without_token_proxied_requests_are_rejected(tmp_path: Path) -> None:
    with TestClient(create_app(build_services(tmp_path)), **LOCAL) as client:
        assert client.get("/api/reports").status_code == 200
        assert client.get("/api/reports", headers={"X-Forwarded-For": "8.8.8.8"}).status_code == 401
        assert client.get("/api/reports", headers={"Host": "abc.ngrok-free.app"}).status_code == 401
    with TestClient(create_app(build_services(tmp_path)), base_url="http://127.0.0.1", client=("203.0.113.9", 1)) as remote:
        assert remote.get("/api/reports").status_code == 401


def test_token_login_flow(tmp_path: Path) -> None:
    app = create_app(build_services(tmp_path, app_token=TOKEN))
    with TestClient(app, base_url="https://scriptmax.example", client=("203.0.113.9", 1)) as client:
        assert client.get("/api/reports").status_code == 401
        assert client.get("/api/config").json()["authenticated"] is False
        assert client.post("/api/login", json={"token": "errado"}).status_code == 401
        login = client.post("/api/login", json={"token": TOKEN})
        assert login.status_code == 204
        cookie = login.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=strict" in cookie and "secure" in cookie
        assert client.get("/api/reports").status_code == 200


def test_login_is_rate_limited(tmp_path: Path) -> None:
    app = create_app(build_services(tmp_path, app_token=TOKEN))
    with TestClient(app, **LOCAL) as client:
        statuses = [client.post("/api/login", json={"token": "x"}).status_code for _ in range(11)]
    assert statuses[:10] == [401] * 10
    assert statuses[10] == 429
