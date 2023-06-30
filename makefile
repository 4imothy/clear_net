CC := gcc
CFLAGS := -Wall -Wextra -pedantic
FORMAT := clang-format
LIB_FILE := clear_net.h
EXAMPLE_DIR := examples
EXAMPLE_FILES := $(shell find $(EXAMPLE_DIR) -name '*.c')
BENCH_FILE := ./bench/bench.py

all: clear_net

clear_net: $(LIB_FILE)

xor: clear_net $(EXAMPLE_FILES)
	$(CC) $(CFLAGS) -o $@ $(EXAMPLE_DIR)/xor.c

run_xor: xor
	./xor

adder: clear_net $(EXAMPLE_FILES)
	$(CC) $(CFLAGS) -o $@ $(EXAMPLE_DIR)/full_adder.c

run_adder: adder
	./adder

iris: clear_net.h $(EXAMPLE_FILES)
	$(CC) $(CFLAGS) -o $@ $(EXAMPLE_DIR)/iris.c

run_iris: iris
	./iris

run_bench: iris adder xor
	python3 $(BENCH_FILE)

FORMAT_STYLE = -style="{BasedOnStyle: llvm, IndentWidth: 4, TabWidth: 4, UseTab: Never}"
format:
	$(FORMAT) $(FORMAT_STYLE) -i $(EXAMPLE_FILES) $(LIB_FILE)

clean:
	rm example
