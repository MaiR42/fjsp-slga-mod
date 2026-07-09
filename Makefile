CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2

SRC = rl.c fjsp.c ga.c main.c
BIN = slga_mod

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $(BIN) $(SRC) -lm

clean:
	rm -f $(BIN)

.PHONY: all clean