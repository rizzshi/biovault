.PHONY: install baseline dp sweep demo test clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt

baseline:
	$(PYTHON) run_simulation.py --no-dp --rounds 5

dp:
	$(PYTHON) run_simulation.py --dp --epsilon 3.0 --rounds 3

sweep:
	$(PYTHON) sweep.py

demo:
	$(PYTHON) app.py

test:
	$(PYTHON) dp.py
	$(PYTHON) model.py
	$(PYTHON) metrics.py
