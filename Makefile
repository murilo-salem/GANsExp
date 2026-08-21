PYTHON ?= python3

.PHONY: test config-check dry-run

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

config-check:
	PYTHONPATH=src $(PYTHON) scripts/abc_run.py --config configs/safras/2324_multisafra.toml --check-inputs --dry-run

dry-run:
	PYTHONPATH=src $(PYTHON) scripts/abc_run.py --config configs/safras/2324_multisafra.toml --dry-run
