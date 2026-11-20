all: doc

.PHONY: doc

doc:
	pdoc --docformat numpy -o ./docs/_build -t ./pdoc_templates/ src/fennol/