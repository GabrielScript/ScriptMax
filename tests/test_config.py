import pytest

from scriptmax.config import ConfigError, load_settings


@pytest.fixture
def base_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    # Valores explícitos: load_dotenv não sobrescreve variáveis já definidas.
    for name, value in {
        "GROQ_API_KEY": "groq", "DEEPSEEK_API_KEY": "deepseek", "FFMPEG_PATH": "ffmpeg",
        "HOST": "127.0.0.1", "APP_TOKEN": "", "BEHIND_PROXY": "",
    }.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


def test_behind_proxy_disabled_by_default(base_env: pytest.MonkeyPatch) -> None:
    assert load_settings().behind_proxy is False


@pytest.mark.parametrize("value", ["1", "true", "YES"])
def test_behind_proxy_enabled(base_env: pytest.MonkeyPatch, value: str) -> None:
    base_env.setenv("BEHIND_PROXY", value)
    assert load_settings().behind_proxy is True


def test_public_host_requires_app_token(base_env: pytest.MonkeyPatch) -> None:
    base_env.setenv("HOST", "0.0.0.0")
    with pytest.raises(ConfigError):
        load_settings()
