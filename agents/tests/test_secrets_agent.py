from __future__ import annotations

from agents.tools.secrets_tool import shannon_entropy, SecretsTool


def test_shannon_entropy_low_for_repetitive_string():
    assert shannon_entropy("aaaaaaaaaa") < 1.0


def test_shannon_entropy_high_for_random_looking_string():
    assert shannon_entropy("Xk9#mQ2vLp8$wRt5nZq3") > 4.0


def test_heuristic_detects_aws_key(tmp_path):
    (tmp_path / "config.py").write_text('API_KEY = "AKIAIOSFODNN7EXAMPLE"\n')
    findings = SecretsTool.scan_with_heuristic(tmp_path)
    assert any(f.finding_type.startswith("AWS Access Key") for f in findings)


def test_heuristic_detects_prefixed_password_variable(tmp_path):
    # Regresión del bug real encontrado al probar esto: \bpassword\b no
    # coincidía con 'mi_password' porque '_' cuenta como carácter de palabra.
    (tmp_path / "config.py").write_text('mi_password = "Xk9#mQ2vLp8$wRt5nZq3"\n')
    findings = SecretsTool.scan_with_heuristic(tmp_path)
    assert any("password" in f.finding_type for f in findings)


def test_heuristic_ignores_normal_low_entropy_strings(tmp_path):
    (tmp_path / "config.py").write_text('normal_var = "hello world"\npassword = "short"\n')
    findings = SecretsTool.scan_with_heuristic(tmp_path)
    assert findings == []


def test_heuristic_detects_private_key_header(tmp_path):
    (tmp_path / "key.pem").write_text("-----BEGIN RSA PRIVATE KEY-----\nMIIB...\n")
    findings = SecretsTool.scan_with_heuristic(tmp_path, extensions=(".pem",))
    assert any("PEM" in f.finding_type for f in findings)


def test_heuristic_skips_git_directory(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config.py").write_text('API_KEY = "AKIAIOSFODNN7EXAMPLE"\n')
    findings = SecretsTool.scan_with_heuristic(tmp_path)
    assert findings == []
