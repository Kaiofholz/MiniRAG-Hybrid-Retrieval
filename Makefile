.PHONY: test demo compile check

test:
	python -m pytest

demo:
	PYTHONPATH=. python examples/offline_minirag_demo.py

compile:
	python -m py_compile minirag/*.py

check: demo test compile
