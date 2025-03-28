.ONESHELL: # Applies to every target in the file!

PYTHON_VERSION ?= $(shell python3 -c "import sys;print('{}.{}'.format(*sys.version_info[:2]))")

# name
.quantum_nn:
	@echo "PYTHON_VERSION: $(PYTHON_VERSION)"
	python$(PYTHON_VERSION) -m venv .quantum_nn
	. .quantum_nn/bin/activate; .quantum_nn/bin/pip$(PYTHON_VERSION) install --upgrade pip$(PYTHON_VERSION) ; .quantum_nn/bin/pip$(PYTHON_VERSION) install -e .[dev,test]

neuralnet: .quantum_nn

clean: .quantum_nn
	rm -rf .quantum_nn
