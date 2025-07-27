iris:
	poetry run python -m agents.iris

iris-api:
	uvicorn iris_backend:app --host 0.0.0.0 --port 8000 --reload