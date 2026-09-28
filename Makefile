.PHONY: setup data test evaluate train serve report recommend clean

setup:
	python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt

data:
	.venv/bin/python scripts/fetch_movielens.py

test:
	.venv/bin/python -m pytest tests -q

evaluate:
	.venv/bin/python scripts/evaluate.py $(if $(DATA),--data $(DATA))

train:
	.venv/bin/python scripts/train.py $(if $(DATA),--data $(DATA))

serve:
	.venv/bin/python scripts/serve.py --reload

# make report EXPORT=~/Downloads/letterboxd-export.zip
report:
	cd src && ../.venv/bin/python -m lbxd.cli $(EXPORT)

# make recommend EXPORT=~/Downloads/letterboxd-export.zip
recommend:
	.venv/bin/python scripts/recommend.py $(EXPORT)

clean:
	rm -rf .pytest_cache **/__pycache__
