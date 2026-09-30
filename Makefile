.PHONY: setup test lint typecheck dashboard seed eval ci-local label clean

VENV=.venv
PY=$(VENV)/bin/python
export PIP_NO_CACHE_DIR=1
export PYTHONPATH=src

setup:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -q -r requirements.txt

test:
	$(PY) -m pytest -q

lint:
	$(VENV)/bin/ruff check src tests bench/scripts dashboard
	$(VENV)/bin/ruff format --check src tests bench/scripts dashboard || true

typecheck:
	$(PY) -m py_compile $$(find src tests bench/scripts dashboard -name '*.py')

dashboard:
	$(VENV)/bin/streamlit run dashboard/app.py --server.port 8501

seed:
	$(PY) bench/scripts/seed.py

eval:
	$(PY) -m llmeval.runner --dataset calibration --judge stub

ci-local:
	$(PY) bench/scripts/ci_gate.py

label:
	@echo "Run: make dashboard, then open the 'Label' page."

clean:
	rm -rf $(VENV) llmeval.db
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
