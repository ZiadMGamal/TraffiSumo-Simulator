.PHONY: install train evaluate test api frontend docker mock-train

ENV_MODE ?= auto
export ENV_MODE

install:
	pip install -r requirements-dev.txt
	cd frontend && npm install

train:
	python scripts/train.py --algorithm dqn --episodes 100

evaluate:
	python scripts/evaluate.py --episodes 10

benchmark:
	python scripts/benchmark.py --episodes 5

test:
	pytest tests/ -v

api:
	python -m backend.main

frontend:
	cd frontend && npm run dev

docker:
	docker compose up --build

mock-train:
	ENV_MODE=mock python scripts/train.py --algorithm dqn --episodes 10
