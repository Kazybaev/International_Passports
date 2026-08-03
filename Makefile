.PHONY: setup run test docker
setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements-dev.txt
run:
	.venv/bin/streamlit run app.py
test:
	.venv/bin/pytest -q
docker:
	docker compose up --build

