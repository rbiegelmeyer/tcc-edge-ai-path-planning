CC     = gcc
CFLAGS = --std=c11 -pedantic -Wall -Wuninitialized

all: compile

compile: main

main: main.c
	$(CC) $(CFLAGS) main.c -o astar -lm

# Linux / WSL2: janela SDL2 — instale com: sudo apt install libsdl2-dev
visualize: main.c
	$(CC) $(CFLAGS) -DVISUALIZE main.c -o astar_vis $(shell sdl2-config --cflags --libs) -lm

test:
	./astar
