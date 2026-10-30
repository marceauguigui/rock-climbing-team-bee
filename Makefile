.PHONY: sim dashboard test deploy sync stop

WALL ?= N2
START ?= hang
VARIANT ?= classic
HOST ?= 127.0.0.1

sim:
	$(MAKE) -C src/microban sim WALL=$(WALL) START=$(START) VARIANT=$(VARIANT) MOVES=$(MOVES)

dashboard:
	$(MAKE) -C src/microban dashboard HOST=$(HOST)

test:
	uv run python -m unittest src/moves/test_moves.py src/microban/src/moves/test_climb.py

deploy:
	$(MAKE) -C src/microban deploy HOST=$(HOST)

sync:
	$(MAKE) -C src/microban sync HOST=$(HOST)

stop:
	$(MAKE) -C src/microban stop HOST=$(HOST)
