# API reference

For whoever is building the front end. The service is FastAPI, so the schema is
also live and interactive at **`/docs`** once it's running — that page is
generated from the code and can't drift out of date, so trust it over this file
if they ever disagree.

```bash
make serve            # http://localhost:8000
```

CORS is preconfigured for `localhost:3000` and `localhost:5173`. Any other
origin needs `ALLOWED_ORIGINS` set (comma-separated) or the browser will block
every response.

---

## `GET /health`

Call this first. The service boots even when its model is missing, so "running"
and "able to answer" are different questions.

```json
{ "ready": true, "detail": "ready", "catalog_films": 87585,
  "model_films": 43884, "nameable_films": 43884, "tmdb_enabled": true }
```

`ready: false` means every other endpoint returns **503**, and `detail` says
why. `tmdb_enabled: false` means posters and overviews come back `null` — worth
checking before you design around them.

`nameable_films` is the one to watch on a deployment. It counts films the model
knows *and* the catalog can name. If it is far below `model_films`, the server
was pointed at a model and a catalog built from different MovieLens versions:
nothing errors, but recommendations silently come only from the overlap. Equal
numbers mean it is configured correctly.

---

## `POST /recommendations`

The main endpoint. `multipart/form-data`, one field:

| field | type | notes |
|---|---|---|
| `export` | file | the `.zip` from Letterboxd → Settings → Import & Export |

Query parameters, all optional:

| param | default | range | what it does |
|---|---|---|---|
| `n` | 20 | 1–100 | how many films |
| `damping` | 0.15 | 0–1 | 0 favours well-known films, 0.3 digs into the tail |
| `min_support` | 2 | 1–20 | how many of their films must back a recommendation |
| `strict_matching` | false | | refuse uncertain title matches as input |
| `hide_watchlist` | false | | also exclude films already on their watchlist |
| `reject` | — | repeatable | MovieLens ids to drop (see the confirmation flow) |
| `metadata` | true | | set false to skip TMDB and return faster |

```js
const form = new FormData();
form.append("export", file);

const res = await fetch(`${API}/recommendations?n=20&damping=0.15`, {
  method: "POST",
  body: form,
});
const data = await res.json();
```

### Response

```json
{
  "profile":        { "watched": 1203, "rated": 847, "watchlist": 42 },
  "usable_ratings": 812,
  "excluded":       1203,
  "used_fallback":  false,
  "tmdb_enabled":   true,
  "films": [
    {
      "rank": 1,
      "score": 0.834,
      "movie_id": 1237,
      "title": "Seventh Seal, The (Sjunde inseglet, Det)",
      "year": 1957,
      "tmdb_id": 490,
      "imdb_id": "tt0050976",
      "genres": ["Drama"],
      "poster_url": "https://image.tmdb.org/t/p/w500/....jpg",
      "backdrop_url": "https://image.tmdb.org/t/p/w1280/....jpg",
      "overview": "A man seeks answers about life, death, and the existence of God...",
      "runtime": 96,
      "director": "Ingmar Bergman",
      "tmdb_rating": 8.1,
      "release_date": "1957-02-16"
    }
  ]
}
```

### Three things to design around

**`used_fallback: true` is not an error.** It means the user had fewer than 5
ratings we could match, so these are simply popular films rather than
personalised ones. Say so in the UI — "we need a few more ratings to
personalise this" — rather than presenting them as recommendations.

**Every TMDB field can be `null`.** Not missing, `null`: when TMDB is
unconfigured, unreachable, or has nothing for that film. The keys are always
present. Titles, years and genres come from MovieLens and are always there, so
a card must render from those alone with a placeholder where the poster goes.

**MovieLens titles are ugly.** `"Seventh Seal, The (Sjunde inseglet, Det)"` —
trailing articles and parenthesised original titles. The API returns them
as-is. Cleaning them up for display is a two-line change on the server; say the
word rather than writing a regex on the client.

---

## `POST /match` — the confirmation step

Same upload, no recommendations. Use it to ask the user about titles we matched
but aren't confident in.

This matters more than it looks. Titles are matched fuzzily, and one wrong
match drags that film's entire neighbourhood up the ranking — a mis-matched
horror film quietly turns the whole list into horror. Confirming the handful we
flag is the cheapest quality win available.

```json
{
  "profile":      { "watched": 1203, "rated": 847, "watchlist": 42 },
  "matched":      831,
  "unmatched":    16,
  "match_rate":   0.981,
  "uncertain": [
    { "name": "Cure", "year": 1997,
      "matched_title": "Cure (Kyua)", "matched_year": 1997,
      "movie_id": 11, "tier": "fuzzy" }
  ],
  "unmatched_titles": ["A Film That Does Not Exist"]
}
```

`tier` is either `fuzzy` (the title is a near-miss) or `year_far` (the title is
exact but the years disagree by more than one — common for films with a
contested release year, like a festival premiere years before general release).

`uncertain` is usually a handful of entries out of hundreds — show them as
"Did you mean…?" cards. For each one the user rejects, pass its `movie_id` to
`/recommendations`:

```
POST /recommendations?reject=11&reject=482
```

**The user is rejecting a film, not a row.** If their export has two titles that
both matched to `movie_id` 11, rejecting it drops both. That's intended.

`unmatched_titles` is capped at 100 entries; `unmatched` is the true count.

---

## Errors

| status | when | what to show |
|---|---|---|
| **400** | not a zip, empty, or no Letterboxd CSVs inside | `detail` is written for end users — display it |
| **413** | over 25 MB, or a zip that expands absurdly | "that doesn't look like a Letterboxd export" |
| **422** | a query parameter is out of range | a bug in the caller; log it |
| **503** | model or catalog not loaded | "service starting up" — poll `/health` |

All errors are `{"detail": "..."}`. The 400 messages are deliberately
user-facing, e.g. *"expected the .zip from Letterboxd → Settings → Import &
Export. If you unzipped it, upload the original archive."*

---

## Configuration

Settings come from the environment, and from a `.env` file in the project root
if one exists (real environment variables win). A `.env` is the reliable place
for `TMDB_API_KEY`: `~/.zshrc` is only read by *interactive* shells, so a key
exported there is invisible to anything launched by a script or supervisor.

```
TMDB_API_KEY=your-key-here
```


| variable | default | |
|---|---|---|
| `MODEL_PATH` | `data/processed/model.npz` | from `scripts/train.py` |
| `CATALOG_DIR` | `data/raw/ml-latest-small` | must match what the model trained on |
| `TMDB_API_KEY` | — | v3 key, from TMDB → Settings → API |
| `TMDB_READ_TOKEN` | — | v4 token; use either, not both |
| `TMDB_CACHE_PATH` | `data/processed/tmdb-cache.sqlite` | safe to delete |
| `ALLOWED_ORIGINS` | `localhost:3000,localhost:5173` | comma-separated |

TMDB metadata is cached on disk for 30 days, including the misses. The first
request for a film costs a round trip; every one after that is free, so don't
design around metadata being slow.

---

## Notes for planning

- **Uploads are stateless.** `/match` and `/recommendations` each take the file
  independently — there's no session to keep. An export is a few hundred KB, so
  uploading twice is cheaper than running a session store. If that ever becomes
  the bottleneck, it's worth revisiting, but it won't at this scale.
- **`score` is comparable within one response, not across them.** Don't show it
  as a percentage or a star rating. Rank order is the meaningful part.
- **A cold request with metadata takes a beat** — up to 20 TMDB fetches, 8 at a
  time. Show a loading state; subsequent requests hit the cache.
