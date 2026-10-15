from types import SimpleNamespace
import sys

import pytest

import run_backend


@pytest.mark.parametrize("arguments, port", [([], 8001), (["--port", "18001"], 18001)])
def test_launcher_uses_python_api_without_reload(monkeypatch, tmp_path, arguments, port):
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda *args, **kwargs: calls.append((args, kwargs))))

    assert run_backend.main(arguments) == 0
    assert calls == [(("app.main:app",), {
        "host": "127.0.0.1",
        "port": port,
        "log_level": "info",
        "reload": False,
        "workers": 1,
    })]
    assert run_backend.Path.cwd() == run_backend.Path(run_backend.__file__).resolve().parent


@pytest.mark.parametrize("value", ["0", "65536", "invalid"])
def test_launcher_rejects_invalid_port(value):
    with pytest.raises(SystemExit) as error:
        run_backend.main(["--port", value])
    assert error.value.code == 2


def test_launcher_reports_missing_uvicorn(monkeypatch, capsys):
    monkeypatch.chdir(run_backend.Path.cwd())
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    assert run_backend.main([]) == 1
    assert "Uvicorn is missing" in capsys.readouterr().err
