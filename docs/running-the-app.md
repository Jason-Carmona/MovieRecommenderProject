# Running the app

Upload your Letterboxd export and get back films you haven't seen. This takes about 10 minutes the first time, mostly downloading and training. After that, starting it takes one command.

You need Python 3.11 or newer and `make`, on Linux or macOS.

## 1. One-time setup

Get the code and move into the project folder:

```bash
git clone https://github.com/Jason-Carmona/MovieRecommenderProject.git
cd MovieRecommenderProject
```

Then:

```bash
make setup                                    # creates .venv and installs dependencies
.venv/bin/python scripts/fetch_movielens.py --full   # MovieLens ml-32m (~250 MB) into data/raw/
make train DATA=data/raw/ml-32m               # trains the model, about 3 minutes
```

Use the commands as written. Plain `make data` and `make train` use a tiny
sample and generated data, which give poor recommendations.

If the download fails with a certificate error, download `ml-32m.zip` from
https://grouplens.org/datasets/movielens/ in a browser and unzip it into `data/raw/`.

## 2. Posters (optional, but much nicer)

Without this step you get titles and years on grey boxes instead of posters.

Posters, directors and runtimes come from TMDB (The Movie Database), which needs a free API key.
With a key, the app also filters TV series and short films out of the list.

**Get a key:**

1. Sign up at https://www.themoviedb.org/signup and confirm your email.
2. Click your avatar, then **Settings → API**, then **Create** (or "Request an API key").
3. Pick **Developer**, accept the terms, and fill in the form. For a personal project, anything honest is fine, e.g. type "Personal",
   name "letterboxd-recs", URL `http://localhost:8000`, and a one-line summary.
   It's approved instantly.
4. Copy the **API Key** (the short 32-character one), not the long "API Read Access Token".
   The long token works too, but goes in `TMDB_READ_TOKEN` instead. Use one, not both.

**Give it to the app.** Create a file called `.env` in the project folder:

```bash
echo "TMDB_API_KEY=paste-your-key-here" > .env
```

`.env` is gitignored, so the key never gets committed. Don't share it or paste it anywhere public.
If the server is already running, restart it so it picks up the key.

**Check the key works** (should print `200`; `401` means the key is wrong):

```bash
curl -s -o /dev/null -w "%{http_code}\n" "https://api.themoviedb.org/3/movie/550?api_key=paste-your-key-here"
```

## 3. Start it

```bash
CATALOG_DIR=data/raw/ml-32m make serve
```

Leave that running and open **http://localhost:8000**.

`CATALOG_DIR` has to point at the same MovieLens set the model was trained on. If you leave it off, the app still runs but picks from only about a fifth of the films.

To check it's healthy, open http://localhost:8000/health. You want `"ready": true`. If you set a key, you also want `"tmdb_enabled": true`.

## 4. Get your recommendations

1. On Letterboxd go to **Settings → Import & Export → Export your data**.
2. On the page, choose the `.zip` you downloaded. Don't unzip it.
3. Click **Recommend**. The first run with posters takes a few seconds; later runs are faster.

If it says it doesn't have enough ratings to personalise, rate a few more films on Letterboxd and export again.

Stop the server with `Ctrl+C`.
