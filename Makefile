.PHONY: test lint format docker api clean

test:
\tpytest src/ tests/ \
\t--ignore=src/llm/streamlit_ui.py \
\t--ignore=src/simulation/bullwhip_visualizer.py \
\t--ignore=src/optimization/dp_visualizer.py

lint:
\truff check src tests

format:
\tblack src tests

docker:
\tdocker build -t geo-aware-mro .

api:
\tuvicorn src.api.main:app --reload

clean:
\trm -rf .pytest_cache
\trm -rf htmlcov
\trm -rf __pycache__
