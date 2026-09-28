import os

from lbxd.api import load_dotenv


def test_reads_keys_and_ignores_noise(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "\n".join([
            "# a comment",
            "",
            'TMDB_API_KEY="abc123"',
            "export ALLOWED_ORIGINS=https://example.com",
            "QUOTED='single'",
            "malformed line without equals",
            "WITH_EQUALS=a=b=c",
        ])
    )
    for key in ("TMDB_API_KEY", "ALLOWED_ORIGINS", "QUOTED", "WITH_EQUALS"):
        monkeypatch.delenv(key, raising=False)

    load_dotenv(env)

    assert os.environ["TMDB_API_KEY"] == "abc123"
    assert os.environ["ALLOWED_ORIGINS"] == "https://example.com"
    assert os.environ["QUOTED"] == "single"
    assert os.environ["WITH_EQUALS"] == "a=b=c"      # only the first = splits


def test_the_real_environment_wins(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("TMDB_API_KEY=from-file")
    monkeypatch.setenv("TMDB_API_KEY", "from-shell")

    load_dotenv(env)
    assert os.environ["TMDB_API_KEY"] == "from-shell"


def test_a_missing_file_is_not_an_error(tmp_path):
    load_dotenv(tmp_path / "nope.env")
