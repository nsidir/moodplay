# moodplay - run `make` or `make help` for the list of targets.
# Recipes avoid shell-specific syntax so they work on Linux and from Git Bash, PowerShell or cmd on Windows.
# Override any variable on the command line, e.g.  make als RANK=64 ALPHA=40

UV      ?= uv
PY      := $(UV) run python
CLI     := $(UV) run moodplay
MANAGE  := $(PY) web/manage.py

# ---- Last.fm crawl (needs LASTFM_API_KEY in .env) ----
SEED_PAGES   ?= 5        # pages of 100 tracks per seed mood tag
TRACKS       ?= 20000    # most popular catalogue tracks to tag
RATE         ?= 4        # requests/second (Last.fm asks for <= 5)

# ---- data / index ----
CATALOG      ?= 60000
TOP_TAGS     ?= 25
MIN_TAG      ?= 15
SPARK_MEM    ?= 3g

# ---- ALS (implicit) ----
RANK         ?= 48
ITERS        ?= 15
REG          ?= 0.1
ALPHA        ?= 20
HOLDOUT      ?= 0.1
PLAYS        ?= 0        # 1 = also train ALS on Taste Profile play counts (measured worse, see README)
SEED         ?= 42

# ---- querying ----
Q            ?= chill late night vibes
K            ?= 15
MOOD         ?= lift
MIN_USERS    ?= 20
YEARS        ?=
EXPLAIN      ?=

# ---- web ----
HOST         ?= 127.0.0.1
PORT         ?= 8001

s = $(strip $(1))
DATA_OPTS = --catalog-size $(call s,$(CATALOG)) --top-tags $(call s,$(TOP_TAGS)) \
            --min-tag-tracks $(call s,$(MIN_TAG)) --driver-memory $(call s,$(SPARK_MEM))
ALS_OPTS  = --rank $(call s,$(RANK)) --iters $(call s,$(ITERS)) --reg $(call s,$(REG)) \
            --alpha $(call s,$(ALPHA)) --holdout $(call s,$(HOLDOUT)) --seed $(call s,$(SEED)) \
            --driver-memory $(call s,$(SPARK_MEM)) $(if $(filter 1,$(call s,$(PLAYS))),--plays)
ASK_OPTS  = -k $(call s,$(K)) --mood $(call s,$(MOOD)) --min-users $(call s,$(MIN_USERS)) \
            $(if $(call s,$(YEARS)),--years $(call s,$(YEARS))) $(if $(call s,$(EXPLAIN)),--explain)

.DEFAULT_GOAL := help
.PHONY: help all install fetch db db-check db-info db-pack db-unpack knn audio crawl crawl-seeds data index build als als-lb \
        evaluate ask chat migrate superuser run test check size clean-als clean

help:
	$(info )
	$(info moodplay targets)
	$(info ------------------)
	$(info   make all            install + fetch + db + knn + audio + build + migrate + als + evaluate)
	$(info   make install        uv sync)
	$(info   make fetch          download the extra sources (~45 GB, resumable) into data/raw/downloads)
	$(info   make db             build/extend data/musicdb from every source present (skips what is already in))
	$(info   make db-check       validate musicdb against its schema (types, keys, references))
	$(info   make db-info        tables, partitions, rows and sizes)
	$(info   make db-pack        musicdb -> data/musicdb-v1.tar, one file to copy anywhere)
	$(info   make db-unpack FILE=musicdb-v1.tar   restore it (then `make knn build als`))
	$(info   make knn            kNN tag model on the Last.fm train split + test-split eval  [MIN_TAG])
	$(info   make audio          audio -> mood-tag model (AcousticBrainz features) + test-split eval)
	$(info   make crawl          optional Last.fm API tags into musicdb, cached + resumable  [SEED_PAGES TRACKS RATE])
	$(info   make crawl-seeds    only the seed tag charts (fast, ~10 min))
	$(info   make data           Spark: catalogue + weighted tags       [CATALOG TOP_TAGS MIN_TAG SPARK_MEM])
	$(info   make index          tag embeddings + track vectors)
	$(info   make build          data + index)
	$(info   make als            Spark implicit ALS incl. web feedback  [RANK ITERS REG ALPHA HOLDOUT SEED PLAYS])
	$(info   make als-lb         Spark implicit ALS, ListenBrainz only  [same params])
	$(info   make evaluate       mood queries vs held-out Last.fm test-split tags)
	$(info   make ask Q="..."    one-shot query                         [K MOOD MIN_USERS YEARS EXPLAIN])
	$(info   make chat           interactive CLI)
	$(info   make migrate / superuser / run / test / check              [HOST PORT])
	$(info   make size           disk usage of data and artifacts)
	$(info   make clean-als / clean)
	$(info )
	$(info examples)
	$(info   make crawl TRACKS=5000)
	$(info   make ask Q="rainy sunday, acoustic" YEARS=1990- EXPLAIN=1)
	$(info   make als RANK=64 ALPHA=40)
	$(info )
	@$(UV) --version

# the Last.fm dataset replaces the API crawl, so one pass is enough (`make crawl build` adds API tags on top)
all:
	$(MAKE) install fetch db knn audio build migrate als evaluate

install:
	$(UV) sync

FILE         ?=

fetch:
	bash scripts/fetch_sources.sh data/raw/downloads

db:
	$(CLI) db build

db-check:
	$(CLI) db check

db-info:
	$(CLI) db info

db-pack:
	$(CLI) db pack $(call s,$(FILE))

db-unpack:
	$(CLI) db unpack $(call s,$(FILE))

knn:
	$(CLI) knn --min-tag-tracks $(call s,$(MIN_TAG))

audio:
	$(CLI) audio

crawl:
	$(CLI) crawl --seed-pages $(call s,$(SEED_PAGES)) --tracks $(call s,$(TRACKS)) --rate $(call s,$(RATE))

crawl-seeds:
	$(CLI) crawl --seed-pages $(call s,$(SEED_PAGES)) --tracks 0 --rate $(call s,$(RATE))

data:
	$(CLI) build-data $(DATA_OPTS)

index:
	$(CLI) build-index

build: data index

als: migrate
	$(MANAGE) train_als $(ALS_OPTS)

als-lb:
	$(CLI) train-als $(ALS_OPTS)

evaluate:
	$(CLI) evaluate

ask:
	$(CLI) ask "$(Q)" $(ASK_OPTS)

chat:
	$(CLI) chat $(ASK_OPTS)

migrate:
	$(MANAGE) migrate --verbosity 0

superuser: migrate
	$(MANAGE) createsuperuser

run: migrate
	$(MANAGE) runserver $(call s,$(HOST)):$(call s,$(PORT))

test:
	$(MANAGE) test recs

check:
	$(MANAGE) check
	$(CLI) ask "chill late night vibes" -k 3

size:
	$(PY) -c "import pathlib as p; [print(f'{d:10s} {sum(f.stat().st_size for f in p.Path(d).rglob(chr(42)) if f.is_file())/2**20:8.0f} MB') for d in ('data/musicdb', 'artifacts') if p.Path(d).exists()]"

clean-als:
	$(PY) -c "import pathlib; [p.unlink(missing_ok=True) for p in map(pathlib.Path, ['artifacts/als_items.npy', 'artifacts/als_meta.json'])]"

clean:
	$(PY) -c "import shutil; shutil.rmtree('artifacts', ignore_errors=True)"
