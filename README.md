# MovieRecommenderProject
This project is to create a webapp that generates recommendations for multiple users. Like a recommended list of movies for a movie night with friends.

## letterboxd-recs

Personal film recommendations from your own Letterboxd history. You upload the
export Letterboxd already gives you, and it recommends films you haven't seen.

Status: **serving over HTTP.** Ingest, ID matching, an item-item collaborative
filtering model, an evaluation harness that shows it beating a popularity
baseline, and a JSON API that returns films with TMDB posters attached.

- **[docs/how-it-works.md](docs/how-it-works.md)** — the walkthrough: the maths,
  and why each piece is shaped the way it is.
- **[docs/api.md](docs/api.md)** — the HTTP contract, for whoever is building
  the front end.

## Why the export and not the API

Letterboxd's API has been invite-only for years, and scraping profiles is
against their terms and breaks whenever their markup changes. Every user can
export their own data from Settings → Import & Export, so the app takes that
zip. No credentials, no rate limits, no gray areas.

## The actual hard part

The export identifies films by title and year — no IDs. MovieLens (the ratings
corpus the model trains on) writes titles its own way:

```
Matrix, The (1999)
Amelie (Fabuleux destin d'Amelie Poulain, Le) (2001)
Seven (a.k.a. Se7en) (1995)
Girl with the Dragon Tattoo, The (Man som hatar kvinnor) (2009)
```

`matching.py` reconciles the two: trailing articles, parenthesised alternate
titles, `a.k.a.` markers, diacritics, punctuation (`WALL·E` / `WALL-E`), `&`
vs `and`, and release-year drift. Matches come back tiered so the UI can trust
`EXACT` silently and surface `FUZZY` for confirmation:

| tier | meaning | |
|---|---|---|
| `EXACT` | same normalised title, same year | trusted |
| `YEAR_OFF` | same title, year within ±1 | trusted |
| `TITLE_ONLY` | same title, no year on one side | trusted |
| `YEAR_FAR` | same title, year off by more than 1 | **ask the user** |
| `FUZZY` | near-miss title inside the year window | **ask the user** |

`YEAR_FAR` exists because of a real failure. *Andrei Rublev* was finished in
1966, shown at Cannes in 1969 and released in the USSR in 1971; Letterboxd and
MovieLens picked different years, the title went unmatched, and because an
unmatched film never reaches the exclude list, the model recommended it back to
a user who had already seen it. Refusing an uncertain match is not the safe
option — it just moves the error somewhere less visible.

Fuzzy search is restricted to films released within a year of the target, so
each lookup compares against hundreds of titles instead of ninety thousand.
Once matched, MovieLens `links.csv` hands over TMDB and IMDb ids for metadata.

## The model

Item-item collaborative filtering: if the people who liked *Stalker* also liked
*Solaris*, those films are neighbours, and someone who loved *Stalker* should
see *Solaris*. It uses no genre, cast or plot data at all — only the pattern of
who rated what.

Four steps, each fixing a specific failure of the one before, all derived in
[docs/how-it-works.md](docs/how-it-works.md):

| step | fixes |
|---|---|
| centre ratings per user | generous vs harsh raters reading as disagreement |
| cosine between item columns | — the similarity itself |
| shrink by co-rater count | two obscure films sharing three raters scoring 1.0 |
| keep top-K neighbours | noise, and a 7.6-billion-cell matrix |

Then a popularity damping term at ranking time, because an untuned version
recommends *Shawshank* to everyone — technically defensible, useless to a
Letterboxd user.

## Does it work?

Yes, and the harness proves it rather than asserting it. Each user's history is
split **by time** (holding out their most recent 20%, so the model never sees
the future it's predicting), and the model is scored against a popularity
baseline and a random floor.

**ml-32m — 25.5M ratings, 200,948 users, 43,884 films.** Trains in 152s.

```
random                   P@10=0.0002  R@10=0.0003  NDCG@10=0.0003  cov=0.362  nov=0.296
popularity               P@10=0.0605  R@10=0.0528  NDCG@10=0.0773  cov=0.003  nov=0.971
item-item CF (damp=0)    P@10=0.0883  R@10=0.0835  NDCG@10=0.1165  cov=0.025  nov=0.896
item-item CF (damp=0.2)  P@10=0.0881  R@10=0.0827  NDCG@10=0.1161  cov=0.028  nov=0.888
item-item CF (damp=0.3)  P@10=0.0867  R@10=0.0813  NDCG@10=0.1144  cov=0.030  nov=0.870

best CF beats popularity by +50.7% NDCG@10
```

`cov` is the share of the catalog ever recommended — the column that exposes
the baseline as naming ~130 films forever, against CF's eight times as many.
`nov` is mean scaled log-popularity of the results, so the damping rows show
exactly what obscurity costs in accuracy: here α=0.2 buys 12% more coverage for
0.3% of NDCG.

### The bug this harness caught

An earlier version scored candidates by dividing the weighted similarity sum by
the total similarity — a weighted average instead of a weighted total. It
passed a hundred tests and a synthetic benchmark that reported CF beating
popularity by 51%.

On real data it scored **worse than random**: NDCG 0.0065 against the
baseline's 0.0770. Removing one division took it to 0.1126, a 17× swing.

Dividing produces a *predicted rating*, which is correct for minimising RMSE
and wrong for ranking, because it discards how much evidence backs each score.
The synthetic data could not catch it: at 10% density every candidate had many
neighbours, so the denominator was near-constant and dividing by it changed
nothing. Real data is 0.29% dense. [The full account is in
docs/how-it-works.md §6 and §8](docs/how-it-works.md) — it is the most useful
thing in this repo.

## The API

```bash
make serve                      # http://localhost:8000, docs at /docs
```

| endpoint | |
|---|---|
| `GET /health` | is a model loaded, is TMDB configured |
| `POST /match` | which titles resolved, and which need the user to confirm |
| `POST /recommendations` | the films, with posters, runtimes and directors |

Upload the export zip as multipart; everything else is query parameters. Full
shapes, error codes and the confirmation flow are in
[docs/api.md](docs/api.md), and FastAPI serves a live interactive schema at
`/docs` that cannot drift out of date.

Two behaviours worth knowing before building against it: every TMDB field can
be `null` (unconfigured, unreachable, or nothing on file — the keys are always
present, so cards must render from title and year alone), and
`used_fallback: true` means the user had too little history to personalise, so
those are popular films rather than recommendations.

TMDB metadata is optional. Without `TMDB_API_KEY` the service still returns
rankings, just without posters. With it, responses are cached on disk for 30
days — including the misses, so a film TMDB doesn't have isn't re-requested on
every page view.

## Run it

```bash
make setup
make data                                    # MovieLens into data/raw/
make test                                    # 101 tests, no download needed
make evaluate                                # model vs baselines
make train
make report EXPORT=~/Downloads/letterboxd-export.zip      # match rate
make recommend EXPORT=~/Downloads/letterboxd-export.zip   # the films
make serve                                   # the HTTP API
```

`make report` prints the match rate per section, every uncertain match to
eyeball, and everything that didn't match at all. On a 21-film arthouse export
against ml-32m:

```
matched 21/21 (100.0%)
  confirm: Andrei Rublev (1966) -> Andrei Rublev (Andrey Rublyov) (1969) [year_far]
```

If `make data` fails on an expired certificate, that's grouplens.org — they let
it lapse periodically. Download the zip in a browser into `data/raw/`, or pass
`--insecure` to the fetch script. If you use `--insecure`, verify the download
afterwards: `ml-32m` ships a `checksums.txt`, and matching it confirms you got
the real files even though the transport was unverified.

## Layout

```
src/lbxd/letterboxd.py   parse the export (zip or directory)
src/lbxd/movielens.py    load the MovieLens catalog + TMDB/IMDb links
src/lbxd/matching.py     title -> movieId, tiered by confidence
src/lbxd/ratings.py      the sparse ratings matrix + id/position maps
src/lbxd/cf.py           the model: similarity, scoring, damping
src/lbxd/evaluate.py     temporal split, metrics, baselines
src/lbxd/recommend.py    export -> ranked films
src/lbxd/tmdb.py         posters and credits, cached in SQLite
src/lbxd/api.py          the HTTP service
src/lbxd/synthetic.py    generated data with known structure, for tests
src/lbxd/cli.py          match-rate report
scripts/                 fetch, train, evaluate, recommend, serve
tests/                   101 tests, a few seconds, no network
```

Ingest and matching are standard library only. The model needs numpy and scipy
and nothing else — no torch, no implicit, no surprise 2 GB dependency. The
tests never touch the network: the model is tested against generated data, and
TMDB against a mock transport.

## Known limits

- **Item cold start.** A film nobody has rated has an empty similarity row and
  can never be recommended — which hits new and obscure releases hardest,
  exactly the films a Letterboxd user most wants surfaced.
- **User cold start.** Under 5 matched ratings it falls back to popularity and
  says so rather than pretending.
- **No sense of time.** A rating from 2015 counts as much as one from last week.

## Next

1. **Content fallback** on TMDB keywords/cast/director for films MovieLens has
   never heard of, blended with the CF score — this is what fixes item cold
   start.
2. **Sweep λ and K on ml-32m.** α is tuned; the other two are still at their
   first-guess defaults of 25 and 100.
3. **Web UI** — in progress separately, against the API above.
>>>>>>> 18d5dd9 (Initial Letterboxd movie recommender)
