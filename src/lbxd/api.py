"""HTTP service: upload a Letterboxd export, get films back.

Run it with:

    make serve                  # uvicorn lbxd.api:app --reload

Interactive docs live at http://localhost:8000/docs — FastAPI generates them
from the response models below, which is the real reason those models are
declared as Pydantic classes instead of plain dicts. A front-end developer can
read the schema instead of reading this file.

Three design notes worth understanding before changing anything here:

**The model is loaded once, at startup.** `ItemItemCF.load` reads a similarity
matrix that can be hundreds of megabytes, and `TitleIndex` walks the whole
catalog to build its lookup tables. Doing either per request would add seconds
to every call for no benefit — the artifacts are immutable. They are built in
the lifespan handler and held on `app.state`.

**Blocking work goes to a thread.** This is an async server: while a coroutine
is running, *nothing else on that worker can progress*, including other users'
requests. Parsing a zip, matching a few thousand titles and multiplying a
sparse matrix are all CPU-bound and synchronous, so they run under
`asyncio.to_thread`. TMDB fetches are genuinely I/O-bound and stay on the event
loop, where concurrency actually helps.

**A missing artifact does not stop the app booting.** It starts, reports
`ready: false` on /health, and returns 503 with a usable message. A container
that crash-loops before it can serve a health check is much harder to diagnose
than one that boots and explains itself.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .cf import ItemItemCF
from .letterboxd import Profile, load_export
from .matching import Tier, TitleIndex, match_films
from .movielens import Movie, load_catalog
from .recommend import recommend_for_profile
from .tmdb import MetadataCache, TMDBClient

ROOT = Path(__file__).resolve().parents[2]

# A Letterboxd export is a few hundred KB even for someone with 5,000 films.
# 25 MB is generous; the cap exists so an upload endpoint cannot be used to
# exhaust the server's disk or memory.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
# Compressed CSV expands roughly 5x. 200 MB of *declared* uncompressed size is
# far beyond any real export, and checking it before reading is what stops a zip
# bomb: a 100 KB archive that decompresses to 10 GB.
MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024


def load_dotenv(path: Path) -> None:
    """Read KEY=value lines from a .env file into the environment.

    Twelve lines instead of a dependency. A real .env file needs this much and
    no more: skip blanks and comments, split on the first `=`, strip one layer
    of quotes.

    Values already in the environment win, so an explicit `TMDB_API_KEY=... make
    serve` still overrides the file — the usual precedence, and the one that
    makes a stray .env impossible to be confused by.

    This also exists for a practical reason: `~/.zshrc` is only read by
    *interactive* shells, so a key exported there is invisible to anything
    launched by a script, an editor, or a supervisor. A .env file in the project
    is visible to all of them.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class Settings:
    """Configuration, all overridable by environment variable."""

    model_path: Path = ROOT / "data/processed/model.npz"
    catalog_dir: Path = ROOT / "data/raw/ml-latest-small"
    cache_path: Path = ROOT / "data/processed/tmdb-cache.sqlite"
    tmdb_api_key: str | None = None
    tmdb_read_token: str | None = None
    allowed_origins: list[str] = field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"]
    )

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(ROOT / ".env")
        origins = os.environ.get("ALLOWED_ORIGINS", "")
        return cls(
            model_path=Path(os.environ.get("MODEL_PATH", cls.model_path)),
            catalog_dir=Path(os.environ.get("CATALOG_DIR", cls.catalog_dir)),
            cache_path=Path(os.environ.get("TMDB_CACHE_PATH", cls.cache_path)),
            tmdb_api_key=os.environ.get("TMDB_API_KEY") or None,
            tmdb_read_token=os.environ.get("TMDB_READ_TOKEN") or None,
            allowed_origins=(
                [o.strip() for o in origins.split(",") if o.strip()]
                or ["http://localhost:3000", "http://localhost:5173"]
            ),
        )


# ---- response models (these become the OpenAPI schema) ------------------


class ProfileOut(BaseModel):
    watched: int = Field(description="films marked watched in the export")
    rated: int
    watchlist: int


class FilmOut(BaseModel):
    rank: int
    score: float = Field(
        description="comparable within one response, not across responses"
    )
    movie_id: int = Field(description="MovieLens id")
    title: str = Field(
        description="render this one — TMDB's title when available, otherwise "
                    "the catalog title tidied for reading"
    )
    catalog_title: str = Field(
        description="raw MovieLens title, for debugging a wrong match"
    )
    year: int | None
    tmdb_id: int | None
    imdb_id: str | None
    genres: list[str]
    # Everything below comes from TMDB and is null when TMDB is unconfigured or
    # had nothing for this film. Render accordingly — do not assume a poster.
    poster_url: str | None = None
    backdrop_url: str | None = None
    overview: str | None = None
    runtime: int | None = None
    director: str | None = None
    tmdb_rating: float | None = None
    release_date: str | None = None


class RecommendationsOut(BaseModel):
    profile: ProfileOut
    usable_ratings: int = Field(description="ratings that matched and fed the model")
    excluded: int = Field(description="their own films kept out of the results")
    used_fallback: bool = Field(
        description="true when there was too little history for CF, so these are "
                    "popular films rather than personalised ones"
    )
    tmdb_enabled: bool
    films: list[FilmOut]


class UncertainMatch(BaseModel):
    """A title we matched but are not confident about. Ask the user."""

    name: str = Field(description="what their export said")
    year: int | None
    matched_title: str = Field(description="what we think it is")
    matched_year: int | None
    movie_id: int = Field(description="pass this to /recommendations?reject= to drop it")
    tier: str


class MatchOut(BaseModel):
    profile: ProfileOut
    matched: int
    unmatched: int
    match_rate: float
    uncertain: list[UncertainMatch]
    unmatched_titles: list[str]


class HealthOut(BaseModel):
    ready: bool
    detail: str
    catalog_films: int = 0
    model_films: int = 0
    nameable_films: int = Field(
        0,
        description="films the model knows AND the catalog can name. Far below "
                    "model_films means MODEL_PATH and CATALOG_DIR were built from "
                    "different MovieLens versions — the service still works, but "
                    "it can only ever recommend this many films.",
    )
    tmdb_enabled: bool = False


# ---- upload handling ----------------------------------------------------


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    """Stream an upload to disk, refusing anything oversized.

    Streamed in chunks rather than `await upload.read()`, because reading an
    unbounded body into memory is how a single request takes the process down.
    """
    written = 0
    with dest.open("wb") as out:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    413,
                    f"export is larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB — "
                    "that is not a Letterboxd export",
                )
            out.write(chunk)
    if written == 0:
        raise HTTPException(400, "empty upload")


def _validate_zip(path: Path) -> None:
    """Check the archive before anything reads it.

    The declared uncompressed size is checked first because decompressing is
    what costs memory. A zip bomb is small on disk and enormous in RAM.
    """
    if not zipfile.is_zipfile(path):
        raise HTTPException(
            400,
            "expected the .zip from Letterboxd → Settings → Import & Export. "
            "If you unzipped it, upload the original archive.",
        )
    with zipfile.ZipFile(path) as z:
        total = sum(i.file_size for i in z.infolist())
        if total > MAX_UNCOMPRESSED_BYTES:
            raise HTTPException(413, "archive expands to an implausible size")
        names = {Path(i.filename).stem.lower() for i in z.infolist()}
    if not names & {"watched", "ratings", "diary", "watchlist"}:
        raise HTTPException(
            400,
            "archive has no watched.csv, ratings.csv, diary.csv or watchlist.csv — "
            "is it a Letterboxd export?",
        )


async def _profile_from_upload(upload: UploadFile) -> Profile:
    """Upload -> parsed Profile, via a temp file that is always cleaned up."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "export.zip"
        await _save_upload(upload, path)
        _validate_zip(path)
        try:
            # Parsing is synchronous and can take a moment on a large export, so
            # it does not belong on the event loop.
            return await asyncio.to_thread(load_export, path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc


# ---- the app -----------------------------------------------------------


def _profile_out(profile: Profile) -> ProfileOut:
    return ProfileOut(
        watched=len(profile.watched),
        rated=len(profile.ratings),
        watchlist=len(profile.watchlist),
    )


def _require_ready(request: Request) -> tuple[ItemItemCF, TitleIndex, list[Movie]]:
    state = request.app.state
    if state.model is None or state.catalog is None:
        raise HTTPException(
            503,
            f"model or catalog not loaded ({state.detail}). "
            "Run scripts/train.py and point MODEL_PATH and CATALOG_DIR at the "
            "artifacts.",
        )
    return state.model, state.index, state.catalog


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app.

    A factory rather than a module-level singleton so tests can inject fixture
    paths instead of mutating the environment — the same reason any expensive
    global gets wrapped in one.
    """
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.model = None
        app.state.catalog = None
        app.state.index = None
        app.state.detail = "not loaded"

        try:
            app.state.catalog = load_catalog(settings.catalog_dir)
            app.state.index = TitleIndex(app.state.catalog)
            app.state.model = ItemItemCF.load(settings.model_path)
            app.state.detail = "ready"
        except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
            # Deliberately not fatal. See the module docstring.
            app.state.detail = f"{type(exc).__name__}: {exc}"

        app.state.cache = MetadataCache(settings.cache_path)
        app.state.tmdb = TMDBClient(
            api_key=settings.tmdb_api_key,
            read_token=settings.tmdb_read_token,
            cache=app.state.cache,
        )
        try:
            yield
        finally:
            app.state.cache.close()

    app = FastAPI(
        title="letterboxd-recs",
        version="0.2.0",
        summary="Film recommendations from a Letterboxd export.",
        lifespan=lifespan,
    )
    # Without this, a browser on localhost:5173 cannot read responses from
    # localhost:8000 at all — different port means different origin. Credentials
    # are not allowed because the API has no sessions or cookies to protect.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthOut)
    async def health(request: Request) -> HealthOut:
        """Readiness, plus the one misconfiguration that fails silently.

        A model trained on ml-32m paired with an ml-latest-small catalog does
        not crash — `recommend_for_profile` just blocks every id it cannot name,
        and quietly recommends from the 9,000 films both agree on instead of the
        43,000 the model knows. `nameable_films` is how you notice.
        """
        state = request.app.state
        nameable = 0
        if state.model is not None and state.catalog is not None:
            known = {m.movie_id for m in state.catalog}
            nameable = sum(1 for mid in state.model.item_ids if int(mid) in known)

        return HealthOut(
            ready=state.model is not None and state.catalog is not None,
            detail=state.detail,
            catalog_films=len(state.catalog or []),
            model_films=len(state.model.item_ids) if state.model else 0,
            nameable_films=nameable,
            tmdb_enabled=state.tmdb.enabled,
        )

    @app.post("/match", response_model=MatchOut)
    async def match(
        request: Request, export: UploadFile = File(description="Letterboxd export .zip")
    ) -> MatchOut:
        """Report how well an export resolves, without recommending anything.

        This exists for the confirmation step in the UI. A fuzzy mismatch fed
        into the model drags that film's whole neighbourhood up the ranking, so
        it is worth asking the user about the handful we are unsure of before
        spending a recommendation on a guess.
        """
        _, index, _ = _require_ready(request)
        profile = await _profile_from_upload(export)
        report = await asyncio.to_thread(match_films, profile.ratings, index)

        return MatchOut(
            profile=_profile_out(profile),
            matched=len(report.matched),
            unmatched=len(report.unmatched),
            match_rate=report.rate,
            uncertain=[
                UncertainMatch(
                    name=m.film.name,
                    year=m.film.year,
                    matched_title=m.movie.title,
                    matched_year=m.movie.year,
                    movie_id=m.movie.movie_id,
                    tier=m.tier.name.lower(),
                )
                for m in report.matched
                if m.uncertain
            ],
            unmatched_titles=[f.name for f in report.unmatched[:100]],
        )

    @app.post("/recommendations", response_model=RecommendationsOut)
    async def recommendations(
        request: Request,
        export: UploadFile = File(description="Letterboxd export .zip"),
        n: int = Query(20, ge=1, le=100),
        damping: float = Query(
            0.15, ge=0.0, le=1.0,
            description="popularity damping: 0 favours well-known films, 0.3 digs",
        ),
        min_support: int = Query(2, ge=1, le=20),
        strict_matching: bool = Query(
            False, description="refuse fuzzy title matches as model input"
        ),
        hide_watchlist: bool = Query(
            False, description="also exclude films already on their watchlist"
        ),
        reject: list[int] = Query(
            default=[],
            description="MovieLens ids to drop from the input — the user's answers "
                        "to /match, for titles we matched wrongly",
        ),
        metadata: bool = Query(True, description="attach TMDB posters and details"),
    ) -> RecommendationsOut:
        """Recommend films the user has not seen."""
        model, index, catalog = _require_ready(request)
        profile = await _profile_from_upload(export)

        recs = await asyncio.to_thread(
            recommend_for_profile,
            profile, model, index, catalog,
            n=n,
            popularity_damping=damping,
            min_support=min_support,
            max_tier=Tier.YEAR_OFF if strict_matching else Tier.FUZZY,
            exclude_watchlist=hide_watchlist,
            reject_ids=set(reject) or None,
        )

        client: TMDBClient = request.app.state.tmdb
        extras = {}
        if metadata and client.enabled:
            # The only genuinely I/O-bound step, and the one place where staying
            # on the event loop pays for itself.
            extras = await client.fetch_many(
                [r.movie.tmdb_id for r in recs if r.movie.tmdb_id]
            )

        films = []
        for rec in recs:
            meta = extras.get(rec.movie.tmdb_id) if rec.movie.tmdb_id else None
            films.append(
                FilmOut(
                    rank=rec.rank,
                    score=rec.score,
                    movie_id=rec.movie.movie_id,
                    # MovieLens titles are built for sorting, not reading
                    # ("Jetee, La"). Prefer TMDB's, fall back to a tidied
                    # catalog title, so this field is always presentable.
                    title=(meta.title if meta and meta.title else rec.movie.display),
                    catalog_title=rec.movie.title,
                    year=rec.movie.year,
                    tmdb_id=rec.movie.tmdb_id,
                    imdb_id=rec.movie.imdb_id,
                    genres=list(rec.movie.genres),
                    poster_url=meta.poster_url if meta else None,
                    backdrop_url=meta.backdrop_url if meta else None,
                    overview=meta.overview if meta else None,
                    runtime=meta.runtime if meta else None,
                    director=meta.director if meta else None,
                    tmdb_rating=meta.tmdb_rating if meta else None,
                    release_date=meta.release_date if meta else None,
                )
            )

        return RecommendationsOut(
            profile=_profile_out(profile),
            usable_ratings=recs.matched_ratings,
            excluded=recs.excluded,
            used_fallback=recs.used_fallback,
            tmdb_enabled=client.enabled,
            films=films,
        )

    return app


app = create_app()
