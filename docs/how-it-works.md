# How it works

A walkthrough of the recommender, in the order the code runs. Every design
choice here is a response to a specific way the previous version broke.

---

## 1. The shape of the problem

You have a matrix `R` where `R[u, i]` is the rating user `u` gave film `i`.
Almost every cell is empty. In ml-32m: 200,000 users × 87,000 films = 17
billion cells, 32 million of them filled. **0.2% density.**

The single most important thing to internalise: an empty cell means
**unknown**, not zero, and not "disliked". Treating unknowns as zeros is the
most common way to build a recommender that is quietly and confidently wrong —
it turns "I have never heard of this film" into "I hate this film", and every
downstream number inherits the lie.

This is why the code uses `scipy.sparse` everywhere and why `center_by_user`
touches only stored entries.

## 2. Centering: fixing the rater, not the film

Two people with identical taste:

| | Stalker | Solaris |
|---|---|---|
| alice | 5.0 | 4.5 |
| bob | 3.0 | 2.5 |

Alice rates generously, Bob doesn't. Raw cosine similarity between these two
user vectors reads them as different, when in fact they agree completely about
which film is better. Subtract each user's own mean:

| | Stalker | Solaris |
|---|---|---|
| alice | +0.25 | −0.25 |
| bob | +0.25 | −0.25 |

Now they're identical, which is the truth. A rating now means *"how far above
or below this person's own average"* — a quantity that is comparable between
people.

Cosine similarity computed on user-centered data has a name in the literature:
**adjusted cosine similarity** (Sarwar et al., 2001). It matters more for
Letterboxd than for MovieLens, because rating culture there is openly personal
— some users treat 3 stars as "good", others as "disappointing".

> **Consequence worth knowing.** A user with exactly two ratings always
> produces one positive and one negative deviation, so any two films they rated
> are perfectly anti-correlated. Adjusted cosine needs at least three ratings
> per user to say anything. This bit me while writing the tests; see
> `test_shrinkage_penalises_thin_evidence`.

## 3. Similarity: who agrees about what

Normalise each item column to unit length, and the cosine between two films is
just a dot product:

```
sim(i, j) = Σ_u c[u,i] · c[u,j]        c = centered, column-normalised
```

Users who rated neither contribute 0, so the sum runs only over people who
rated both. One multiplication answers *"do the same people feel the same way
about these two films?"*

Note what is **not** in that formula: genre, cast, director, plot, year. The
model has no idea what a film is *about*. It discovers that *Perfect Blue* and
*Paprika* belong together purely from overlapping audiences — and it also finds
connections no metadata could express, like "films that appeal to people going
through a specific kind of twenties". That is the whole appeal of collaborative
filtering, and also its weakness: it can say nothing at all about a film nobody
has rated yet.

### Why item–item rather than user–user

- Fewer items than users, so the similarity matrix is smaller.
- Item neighbourhoods are **stable**. Your taste shifts month to month; the
  fact that *Stalker* and *Solaris* attract the same crowd does not.
- **A brand new user can be served immediately.** We look up their films in a
  matrix built without them. That is exactly our situation — someone uploads a
  Letterboxd export and expects an answer now, and they are not in MovieLens at
  all. A user–user model would have to be retrained to say anything.

## 4. Shrinkage: cosine has no idea how much evidence it has

Two obscure films rated by the same three people can score a perfect 1.0 by
coincidence. Cosine cannot tell that apart from two films co-rated by ten
thousand people. Without a correction, those flukes dominate every top-K list
and the recommendations turn to noise.

So pull every similarity toward zero in proportion to how little supports it:

```
sim'(i, j) = sim(i, j) · n_ij / (n_ij + λ)        n_ij = users who rated both
```

With **λ = 25**:

| co-raters | fraction kept |
|---|---|
| 3 | 11% |
| 25 | 50% |
| 100 | 80% |
| 250 | 91% |

Well-evidenced neighbourhoods survive nearly intact; coincidences collapse.

**λ is the most useful knob in the model.** Raise it if recommendations look
random; lower it if they look generic.

## 5. Top-K: truncation is denoising

The full similarity matrix is n_items². For ml-32m that is 7.6 billion cells —
it fits nowhere. It's also mostly noise: a film's 5,000th-most-similar film
tells you nothing.

Keeping the K best neighbours per item (K=100) makes the model small, fast,
**and more accurate**. This is one of the rare places where the efficient
choice is also the better one.

We also drop negative similarities. "People who liked A disliked B" is real in
principle, but in practice it's dominated by noise from sparse overlaps.

## 6. Scoring: the user's films vote

```
score(j) = Σ_{i ∈ rated} sim(i, j) · (r_i − r̄_u)
```

Each film you've rated votes for its neighbours, weighted by how much you liked
it *relative to your own average*. Centering here is what lets a bad rating
push films **away**: rate *Crash* 1.5 when your average is 3.5, and its weight
is −2.0, dragging its whole neighbourhood down. An uncentered version could
only ever add enthusiasm, never subtract it.

### The division that cost 17× accuracy

The first version of this divided that sum by the total similarity:

```
             Σ_{i ∈ rated} sim(i, j) · (r_i − r̄_u)      ← don't do this
score(j) = ────────────────────────────────────────
             Σ_{i ∈ rated} |sim(i, j)|  +  ε
```

The reasoning seemed sound: turn the sum into a weighted average so a film
isn't favoured merely for having many neighbours in your history. It is also
what the classic item-based CF papers do.

It was wrong, and measurably so:

| scoring rule | NDCG@10, ml-latest-small |
|---|---|
| normalised (divided) | **0.0065** — worse than random |
| raw weighted sum | **0.1126** — +46% over popularity |

Dividing gives you a **predicted rating**, which is the right quantity if
you're minimising RMSE on held-out ratings — exactly what those papers were
measuring. It is the wrong quantity for **ranking**, because it discards how
much evidence supports each score. A film that thirty of your favourites point
at and a film that one lucky neighbour points at come out identical, and the
second kind is vastly more numerous, so it floods the top of the list. For
top-N recommendation the raw weighted sum is the standard (Deshpande &
Karypis, 2004).

The normalised form is still available as `normalize=True`, because it is
correct for rating prediction and because the contrast is worth reproducing.

> **The part worth internalising.** This bug survived a hundred passing tests
> and a synthetic benchmark that reported the model beating popularity by 51%.
> It died within a minute of meeting real data. §8 explains why the synthetic
> data was constitutionally unable to catch it.

`min_support` guards the same failure from the other side: require at least *n*
of the user's films to neighbour a candidate before it can be recommended. It
can legitimately return an empty list, and that's the honest outcome — see
`test_min_support_can_legitimately_empty_the_list`.

## 7. Popularity damping: the knob that makes it usable

Untuned, this model recommends *The Shawshank Redemption* to everyone. Popular
films accumulate the most similarity edges and drift to the top of every
neighbourhood ranking. Technically defensible; useless to a Letterboxd user,
who has seen it and wants to be told about something they haven't.

```
final(j) = score(j) − α · scale_u · log1p(popularity_j) / max(log1p(popularity))
```

Why the log: the most-rated film in MovieLens has ~100,000 ratings and the
median has ~3. A linear penalty would be one enormous cliff. Counts are roughly
log-normal, so `log1p` spreads them evenly; rescaling to 0–1 makes α mean the
same thing on any dataset.

Why subtract rather than divide: scores can be negative, and dividing a
negative number by popularity inverts the ordering.

Why `scale_u`, the largest absolute score in *this user's* results: raw
weighted sums have no fixed scale. Someone with 2,000 ratings produces scores
an order of magnitude larger than someone with 50, so a constant subtraction
would be crushing for one and invisible for the other. Anchoring to the top
score makes α mean the same thing for everybody — *"penalise the most-rated
film by up to this fraction of the best score"*.

This is a direct consequence of §6, and it's worth seeing why: under the old
normalised scoring, scores **were** predicted ratings on a fixed 0–5 scale, so
no rescaling was needed. Removing the division changed the units of the output,
and every downstream constant that assumed those units had to be revisited.
The test that caught it (`test_popularity_damping_shifts_picks_toward_the_tail`)
failed immediately, which is the argument for having it.

α ≈ 0.1–0.3 is the useful range, and it is cheap. On ml-32m, α=0.2 costs 0.3%
NDCG while raising catalog coverage by 12%. This is a **taste knob** — tune it
by looking at output, then confirm with the harness that accuracy hasn't
collapsed.

---

## 8. Evaluation: making the claim falsifiable

Recommendations are dangerously easy to fool yourself about. The output is a
list of plausible film titles, and plausible film titles look good whether the
model learned anything or not. This is why so many recommender projects ship
with no evidence at all.

**The protocol:**

1. Split each user's history **by time**, holding out their most recent 20%.
2. Train on the older ratings only.
3. Ask for 10 recommendations, excluding everything in their training half.
4. Count how many held-out films they actually *liked* (≥ 4.0) turn up.

Step 1 is the one people get wrong. A **random** split leaks the future into
the past — the model sees you loved *Aftersun* in 2023 while predicting what
you watched in 2019. A **temporal** split asks the real question: given
everything you'd seen by some date, what did you watch next?

`test_training_never_sees_the_holdout` exists specifically to catch the leak,
because a leak inflates every metric while the code keeps running.

### The metrics

| metric | question |
|---|---|
| **precision@10** | Of the 10 films we named, how many were good calls? |
| **recall@10** | Of the films they liked, how many did we surface? |
| **NDCG@10** | Same, but a hit at rank 1 counts more than one at rank 10 |

NDCG is the one to lead with, because it's the only one that notices the
difference between a great first recommendation and a great tenth — and users
only look at the top:

```
DCG@k = Σ_{i=1..k} hit_i / log₂(i + 1)
```

A hit at position 1 is worth 1.00; at position 2, 0.63; at position 10, 0.29.
Normalising by the best achievable arrangement gives a 0–1 score comparable
across users with different numbers of relevant films.

> Absolute values always look low. If a user only has 3 relevant held-out
> films, precision@10 cannot exceed 0.3. **Compare rows to each other, never to
> 1.0.**

### The baselines

**Popularity** — recommend the most-rated films to everyone, identically. No
personalisation at all. It is surprisingly hard to beat, because popular films
are popular precisely because most people like them. *A personalised model that
cannot beat this is not doing its job.* Reporting your model's precision with
nothing to compare it against tells the reader nothing.

**Random** — the floor. If you can't beat this, something is wired backwards.

### The two diagnostics that catch degenerate models

Accuracy alone will happily reward a model that's useless:

- **coverage** — what fraction of the catalog ever gets recommended to anyone.
  The popularity baseline scores 0.003 on ml-32m: it names the same handful
  of films forever. A recommender that only knows 200 films isn't much of a
  recommender, however good its precision.
- **novelty** — mean scaled log-popularity of what got recommended; lower is
  more obscure. This is the number that shows damping working: turn α up, watch
  novelty fall, then check what it cost in NDCG.

### Current results

**ml-32m** — 25.5M ratings, 200,948 users, 43,884 films, 2,000 held-out users
scored. Training takes 152 seconds and produces 4.35M similarity edges.

```
random                   P@10=0.0002  R@10=0.0003  NDCG@10=0.0003  cov=0.362  nov=0.296
popularity               P@10=0.0605  R@10=0.0528  NDCG@10=0.0773  cov=0.003  nov=0.971
item-item CF (damp=0)    P@10=0.0883  R@10=0.0835  NDCG@10=0.1165  cov=0.025  nov=0.896
item-item CF (damp=0.1)  P@10=0.0882  R@10=0.0828  NDCG@10=0.1165  cov=0.026  nov=0.893
item-item CF (damp=0.2)  P@10=0.0881  R@10=0.0827  NDCG@10=0.1161  cov=0.028  nov=0.888
item-item CF (damp=0.3)  P@10=0.0867  R@10=0.0813  NDCG@10=0.1144  cov=0.030  nov=0.870

best CF beats popularity by +50.7% NDCG@10
```

Read it in four passes:

1. **CF beats popularity by 51% NDCG.** That is the headline, and the
   comparison — not the 0.1165 — is the claim. Absolute NDCG is capped by how
   many films each user liked in their held-out window.
2. **Popularity beats random by 250×.** The ordering validates the harness
   itself. If popularity had not crushed random, the metrics would be suspect
   before any conclusion about the model.
3. **Coverage: 0.025 vs 0.003.** The baseline's respectable precision comes
   from naming about 130 films, forever. CF names eight times as many. This is
   the column that exposes a model with good accuracy and no usefulness.
4. **Damping is nearly free here.** α=0.2 costs 0.3% NDCG and buys 12% more
   coverage. On the smaller ml-latest-small the same move costs 3.6% — more
   data means more well-evidenced neighbourhoods to fall back on, so there is
   less to lose by declining the obvious answer.

### Why the synthetic benchmark lied

The generated dataset reported CF beating popularity by 51% *while the
normalised scoring of §6 was still in place* — a configuration that turned out
to be worse than random on real data. The numbers were not fabricated; the
dataset simply could not express the failure.

`synthetic.make_ratings` gives 400 users 40 ratings each over 300 films: **10%
dense**. ml-32m is **0.29% dense**. At 10% density nearly every candidate film
has many of the user's ratings as neighbours, so the denominator `Σ|sim|` was
roughly constant across candidates — dividing by a near-constant changes
nothing, and the bug was invisible. Real data has a long tail where that
denominator varies by orders of magnitude, and dividing by it handed the top of
every list to obscure films with a single lucky neighbour.

The lesson is not "synthetic data is useless" — it caught real bugs, and it
still lets the suite run offline in under a second. The lesson is narrower and
more useful: **a synthetic benchmark can only test failures its generator is
capable of producing.** Ours modelled taste structure faithfully and sparsity
not at all, so it validated everything except the thing that mattered. Treat a
green synthetic benchmark as evidence the pipeline runs, never as evidence the
model works.

---

## 9. Tuning, and how to know a difference is real

λ and K sat at first-guess defaults (25 and 100) while the pipeline was built.
Sweeping them on ml-32m against 2,000 held-out users gave a 15-point grid
spanning NDCG 0.1141–0.1212 — a 6% range, with K=100 scoring *worse* than both
K=50 and K=200 at every λ. A shape like that corresponds to no mechanism, so
the first question is not "which won" but "can this table tell configs apart at
all?"

### Standard error, and why pairing changes the answer

The metric is a mean over users, so its precision is bounded by how many users
were scored: `SE = std / sqrt(n)`. Here that is **±0.0039** — and the gap
between the best and default configs is 0.0052, about 1.3 SE. Read that way,
the entire sweep is noise and the "winner" is whichever config drew the luckier
users.

But the two configs were scored on *the same* users, so they can be compared
pairwise:

```
var(b − a) = var(a) + var(b) − 2·cov(a, b)
```

Users differ enormously in how predictable they are, and that variance is
common to both configs — the same people are easy or hard for each. The
covariance term cancels it:

```
winner vs default   +0.0052   paired SE 0.0013   95% CI [+0.0026, +0.0078]   SIGNIFICANT
small  vs default   +0.0024   paired SE 0.0014   95% CI [−0.0005, +0.0052]   within noise
CF     vs popularity +0.0387  paired SE 0.0039                               SIGNIFICANT
```

The SE fell from 0.0039 to 0.0013 and an unreadable result became a 4σ one.
Note the second row: K=50 vs K=100 really *is* noise. Pairing does not make
everything significant — it makes the question answerable.

> The saving comes from covariance, not from pairing as such. For two unrelated
> models `cov` is near zero and the paired SE can be *larger* than either mean's
> own. Pinned down in `test_pairing_does_not_help_for_unrelated_models`.

### The boundary trap

The winner was λ=100, K=200 — the corner of the grid, highest value tested on
both axes. An optimum on a boundary is not an optimum, it is a grid that
stopped too early. Extending it:

| config | NDCG@10 | vs λ=100 K=200 | edges |
|---|---|---|---|
| λ=100 K=200 | 0.1212 | — | 8.7M |
| λ=100 K=400 | 0.1235 | +0.0023, SE 0.0011 — **significant** | 17.3M |
| λ=200 K=200 | 0.1216 | +0.0003, SE 0.0006 — within noise | 8.7M |

So λ genuinely plateaus at 100, while K keeps paying — with each doubling
buying half as much and costing twice the artifact. `scripts/sweep.py` now
warns when its own winner lands on the grid edge.

**Defaults are now λ=100, K=200**: +4.5% NDCG over the old settings for a 49 MB
artifact. K=400 is another +2% for 100 MB, available through
`ItemItemCF.truncate` without refitting, and not worth it as a default.

## 10. What this model still can't do

Worth being honest about, since these are the next things to build:

- **Cold start on items.** A film nobody has rated has an empty similarity row
  and can never be recommended. This hits new and obscure releases hardest —
  exactly the films a Letterboxd user most wants surfaced. Fix: a content-based
  score from TMDB keywords, cast and crew, blended in.
- **Cold start on users.** Under 5 matched ratings, `recommend_for_profile`
  falls back to popularity and says so.
- **No sense of time.** A film you rated in 2015 counts as much as one from
  last week.
- **Popularity bias is damped, not solved.** α is a blunt global constant, not
  a per-user notion of how adventurous someone is.
- **Coverage is still only 0.025.** Even beating the baseline eightfold, the
  model never mentions 97% of the catalog to anyone. That ceiling is set by
  top-K truncation and by the long tail having too few ratings to earn an
  edge — the content fallback in the first bullet is what lifts it.
