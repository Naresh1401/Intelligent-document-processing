.PHONY: install run-api run-ui test docker-up docker-down

install:
	pip install -r requirements.txt

run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	streamlit run ui/app.py

test:
	python tests/test_idp.py

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down
