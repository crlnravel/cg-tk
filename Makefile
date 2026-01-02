.DEFAULT_GOAL := all

.PHONY: all install run clean

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(PY) -m pip
DEPS := $(VENV)/.deps

all: run

$(VENV):
	python -m venv $(VENV)

$(DEPS): requirements.txt | $(VENV)
	$(PIP) install -r requirements.txt
	touch $(DEPS)

install: $(DEPS)

run: $(DEPS)
	$(PY) main.py

clean:
	rm -rf $(VENV)

