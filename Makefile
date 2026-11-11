# Reproduce the experiments and figures.  See README.md.
PY      ?= python3
REPS    ?= 30
LEVELS  := 0 1 2 3 4 5
BOUND   := data/boundaries/surabaya_kecamatan.geojson

.PHONY: all experiments levels benchmark $(addprefix bench-,$(LEVELS)) frontier figures

all: experiments figures

experiments: benchmark frontier

levels:
	$(PY) scripts/calibrate_levels.py

benchmark: $(addprefix bench-,$(LEVELS))

$(addprefix bench-,$(LEVELS)): bench-%:
	$(PY) scripts/run_benchmark.py $* $(REPS)

frontier:
	$(PY) scripts/run_frontier.py

$(BOUND):
	$(PY) scripts/fetch_boundaries.py

figures: $(BOUND)
	$(PY) scripts/make_figures.py
	$(PY) scripts/make_maps.py
	$(PY) scripts/make_framework.py
