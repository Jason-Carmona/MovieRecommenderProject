import io
import os
import zipfile
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from lbxd import api as api_module
from lbxd import synthetic
from lbxd.api import Settings, create_app
from lbxd.cf import ItemItemCF
from lbxd.movielens import load_catalog
from lbxd.tmdb import MetadataCache, TMDBClient

FIXTURES = Path(__file__).parent / "fixtures"

PARASITE = {
    "title": "Parasite", "overview": "Greed and class discrimination.",
    "poster_path": "/p.jpg", "backdrop_path": "/b.jpg", "runtime": 133,
    "vote_average": 8.5, "release_date": "2019-05-30",
    "genres": [{"name": "Thriller"}],
    "credits": {"crew": [{"job": "Director", "name": "Bong Joon-ho"}]},
}

RATINGS = """Date,Name,Year,Letterboxd URI,Rating
2024-01-02,The Matrix,1999,https://boxd.it/1a,5.0
2024-01-03,Amélie,2001,https://boxd.it/2a,4.5
2024-01-04,Se7en,1995,https://boxd.it/3a,4.0
2024-01-05,The Thing,1982,https://boxd.it/4a,4.5
2024-01-06,Parasite,2019,https://boxd.it/5a,5.0
2024-01-07,The Matri,1999,https://boxd.it/6a,3.5
"""

WATCHED = """Date,Name,Year,Letterboxd URI
2024-01-02,The Matrix,1999,https://boxd.it/1a
2024-01-03,Amélie,2001,https://boxd.it/2a
"""


def make_zip(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
    return buffer.getvalue()


def upload(content: bytes, name: str = "letterboxd-export.zip") -> dict:
    return {"export": (name, content, "application/zip")}


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    catalog = load_catalog(FIXTURES / "movielens")
    data = synthetic.make_ratings(n_users=300, n_items=len(catalog), seed=23).to_dataset()
    path = tmp_path_factory.mktemp("model") / "model.npz"
    ItemItemCF.fit(data.matrix, data.item_ids, top_k=6).save(path)
    return path


@pytest.fixture
def client(model_path, tmp_path):
    settings = Settings(
        model_path=model_path,
        catalog_dir=FIXTURES / "movielens",
        cache_path=tmp_path / "cache.sqlite",
        allowed_origins=["http://localhost:5173"],
    )
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def export_zip():
    return make_zip({
        "letterboxd-export/ratings.csv": RATINGS,
        "letterboxd-export/watched.csv": WATCHED,
    })


# ---- readiness ----------------------------------------------------------


def test_health_reports_ready_when_artifacts_loaded(client):
    body = client.get("/health").json()
    assert body["ready"] is True
    assert body["catalog_films"] == 14
    assert body["tmdb_enabled"] is False       # no key configured in tests


def test_a_missing_model_boots_anyway_and_explains_itself(tmp_path, export_zip):
    """A container that crash-loops before serving /health is much harder to
    debug than one that boots and says what is wrong."""
    settings = Settings(
        model_path=tmp_path / "absent.npz",
        catalog_dir=tmp_path / "absent",
        cache_path=tmp_path / "cache.sqlite",
    )
    with TestClient(create_app(settings)) as c:
        health = c.get("/health").json()
        assert health["ready"] is False
        assert "FileNotFoundError" in health["detail"]

        failed = c.post("/recommendations", files=upload(export_zip))
        assert failed.status_code == 503
        assert "train.py" in failed.json()["detail"]


# ---- the happy path -----------------------------------------------------


def test_recommendations_returns_unseen_films(client, export_zip):
    body = client.post("/recommendations?n=5&min_support=1",
                       files=upload(export_zip)).json()

    assert body["profile"] == {"watched": 2, "rated": 6, "watchlist": 0}
    assert body["usable_ratings"] == 6
    assert body["used_fallback"] is False
    assert 0 < len(body["films"]) <= 5

    ranks = [f["rank"] for f in body["films"]]
    assert ranks == sorted(ranks) == list(range(1, len(ranks) + 1))

    seen = {"Matrix, The", "Amelie", "Seven", "Thing, The", "Parasite"}
    assert not {f["title"] for f in body["films"]} & seen

    first = body["films"][0]
    assert first["movie_id"] > 0 and first["title"]
    assert first["poster_url"] is None         # TMDB off: nulls, not absent keys


def test_thin_history_says_so_rather_than_pretending(client):
    thin = make_zip({"ratings.csv":
                     "Date,Name,Year,Letterboxd URI,Rating\n"
                     "2024-01-02,The Matrix,1999,https://boxd.it/1a,5.0\n"})
    body = client.post("/recommendations?n=3", files=upload(thin)).json()

    assert body["used_fallback"] is True
    assert len(body["films"]) > 0


def test_n_is_bounded(client, export_zip):
    assert client.post("/recommendations?n=0", files=upload(export_zip)).status_code == 422
    assert client.post("/recommendations?n=500", files=upload(export_zip)).status_code == 422


def test_damping_reaches_the_model(client, export_zip):
    """Only that the parameter is plumbed through. The fixture catalog is 14
    films with a narrow popularity spread, so the *ordering* does not move here;
    `test_popularity_damping_shifts_picks_toward_the_tail` checks that properly
    against a 150-film model."""
    def scores(damping):
        body = client.post(
            f"/recommendations?n=8&min_support=1&damping={damping}",
            files=upload(export_zip),
        ).json()
        return [f["score"] for f in body["films"]]

    assert scores(0.0) != scores(1.0)


# ---- the confirmation flow ---------------------------------------------


def test_match_surfaces_only_the_uncertain_titles(client, export_zip):
    body = client.post("/match", files=upload(export_zip)).json()

    assert body["matched"] == 6
    assert body["match_rate"] == 1.0
    uncertain = body["uncertain"]
    assert len(uncertain) == 1
    assert uncertain[0]["name"] == "The Matri"
    assert uncertain[0]["matched_title"] == "Matrix, The"
    assert uncertain[0]["tier"] == "fuzzy"
    assert uncertain[0]["movie_id"] == 1


def test_match_lists_titles_it_could_not_resolve(client):
    odd = make_zip({"ratings.csv":
                    "Date,Name,Year,Letterboxd URI,Rating\n"
                    "2024-01-02,A Film That Does Not Exist,2023,https://boxd.it/x,4.0\n"})
    body = client.post("/match", files=upload(odd)).json()

    assert body["unmatched"] == 1
    assert body["unmatched_titles"] == ["A Film That Does Not Exist"]
    assert body["match_rate"] == 0.0


def test_rejecting_a_match_removes_it_from_the_input(client, export_zip):
    kept = client.post("/recommendations?n=8&min_support=1",
                       files=upload(export_zip)).json()
    dropped = client.post("/recommendations?n=8&min_support=1&reject=1&reject=2",
                          files=upload(export_zip)).json()

    assert kept["usable_ratings"] == 6
    # Rejecting id 1 drops *both* rows that matched to it — "The Matrix" and the
    # fuzzy "The Matri" — so six ratings become three, not four. Worth knowing
    # when building the confirmation UI: the user is rejecting a film, not a row.
    assert dropped["usable_ratings"] == 3
    assert not {1, 2} & {f["movie_id"] for f in dropped["films"]}


def test_strict_matching_refuses_fuzzy_input(client, export_zip):
    loose = client.post("/recommendations?n=5&min_support=1",
                        files=upload(export_zip)).json()
    strict = client.post("/recommendations?n=5&min_support=1&strict_matching=true",
                         files=upload(export_zip)).json()

    assert loose["usable_ratings"] == 6
    assert strict["usable_ratings"] == 5


# ---- metadata -----------------------------------------------------------


def test_metadata_is_attached_when_tmdb_is_configured(client, export_zip, tmp_path):
    def handler(request):
        return httpx.Response(200, json=PARASITE)

    client.app.state.tmdb = TMDBClient(
        api_key="k",
        cache=MetadataCache(tmp_path / "meta.sqlite"),
        transport=httpx.MockTransport(handler),
    )
    body = client.post("/recommendations?n=3&min_support=1",
                       files=upload(export_zip)).json()

    assert body["tmdb_enabled"] is True
    first = body["films"][0]
    assert first["poster_url"] == "https://image.tmdb.org/t/p/w500/p.jpg"
    assert first["director"] == "Bong Joon-ho"
    assert first["runtime"] == 133


def test_metadata_can_be_turned_off_per_request(client, export_zip, tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=PARASITE)

    client.app.state.tmdb = TMDBClient(
        api_key="k", cache=MetadataCache(tmp_path / "m.sqlite"),
        transport=httpx.MockTransport(handler),
    )
    body = client.post("/recommendations?n=3&min_support=1&metadata=false",
                       files=upload(export_zip)).json()

    assert calls == []
    assert body["films"][0]["poster_url"] is None


def test_tmdb_being_down_does_not_break_the_recommendations(client, export_zip, tmp_path):
    """The ranking is the product. A poster outage must cost posters, not films."""
    def handler(request):
        raise httpx.ConnectError("tmdb unreachable")

    client.app.state.tmdb = TMDBClient(
        api_key="k", cache=MetadataCache(tmp_path / "m.sqlite"),
        transport=httpx.MockTransport(handler),
    )
    response = client.post("/recommendations?n=3&min_support=1", files=upload(export_zip))

    assert response.status_code == 200
    body = response.json()
    assert len(body["films"]) > 0
    assert all(f["poster_url"] is None for f in body["films"])


# ---- upload validation --------------------------------------------------


def test_a_non_zip_upload_is_rejected_with_a_useful_message(client):
    response = client.post("/recommendations",
                           files=upload(b"Name,Year\nStalker,1979\n", "ratings.csv"))
    assert response.status_code == 400
    assert "Import & Export" in response.json()["detail"]


def test_an_unrelated_zip_is_rejected(client):
    response = client.post("/recommendations",
                           files=upload(make_zip({"notes.txt": "hello"})))
    assert response.status_code == 400
    assert "Letterboxd export" in response.json()["detail"]


def test_an_empty_upload_is_rejected(client):
    assert client.post("/recommendations", files=upload(b"")).status_code == 400


def test_an_oversized_upload_is_refused_before_it_is_read(client, monkeypatch):
    monkeypatch.setattr(api_module, "MAX_UPLOAD_BYTES", 1024)
    # Random titles so the archive does not simply compress back under the cap —
    # the check is on bytes received over the wire, which are compressed bytes.
    rows = "".join(
        f"2024-01-01,{os.urandom(16).hex()},2000,u,4.0\n" for _ in range(500)
    )
    big = make_zip({"ratings.csv": "Date,Name,Year,Letterboxd URI,Rating\n" + rows})
    assert len(big) > 1024

    response = client.post("/recommendations", files=upload(big))
    assert response.status_code == 413


def test_a_zip_bomb_is_refused_on_declared_size(client, monkeypatch):
    """Checked before decompressing, because decompressing is what costs memory."""
    monkeypatch.setattr(api_module, "MAX_UNCOMPRESSED_BYTES", 1024)
    bomb = make_zip({"ratings.csv": "a" * 100_000})     # compresses to almost nothing
    assert len(bomb) < 1024

    response = client.post("/recommendations", files=upload(bomb))
    assert response.status_code == 413
    assert "implausible" in response.json()["detail"]


# ---- browser access -----------------------------------------------------


def test_cors_allows_the_configured_front_end_origin(client, export_zip):
    response = client.post(
        "/recommendations?n=2&min_support=1",
        files=upload(export_zip),
        headers={"Origin": "http://localhost:5173"},
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_the_schema_is_published_for_the_front_end(client):
    schema = client.get("/openapi.json").json()
    assert "/recommendations" in schema["paths"]
    assert "/match" in schema["paths"]


def test_health_exposes_a_model_catalog_mismatch(tmp_path_factory, tmp_path):
    """The misconfiguration that degrades silently: a model trained on one
    MovieLens version served with another's catalog still answers, but only from
    the overlap. nameable_films is how an operator spots it."""
    wide = synthetic.make_ratings(n_users=200, n_items=300, seed=29).to_dataset()
    path = tmp_path_factory.mktemp("wide") / "model.npz"
    ItemItemCF.fit(wide.matrix, wide.item_ids, top_k=5).save(path)

    settings = Settings(
        model_path=path,
        catalog_dir=FIXTURES / "movielens",       # only 14 of the model's 300 ids
        cache_path=tmp_path / "c.sqlite",
    )
    with TestClient(create_app(settings)) as c:
        body = c.get("/health").json()

    assert body["ready"] is True
    assert body["model_films"] == 300
    assert body["catalog_films"] == 14
    assert body["nameable_films"] == 14           # the signal
