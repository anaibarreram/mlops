PYTHON=python

.PHONY: train experiments test prepare
prepare:
	$(PYTHON) scripts/prepare_data.py
train:
	$(PYTHON) -m churn.modeling.train
experiments:
	$(PYTHON) scripts/run_experiments.py
test:
	$(PYTHON) -m pytest -q
