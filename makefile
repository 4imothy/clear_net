CC := gcc
CFLAGS := -Wall -Wextra -pedantic
FORMAT := clang-format
LIB_FILE := clear_net.h
EXAMPLE_DIR := examples
EXAMPLE_FILES := $(wildcard $(EXAMPLE_DIR)/*.c)
BENCH_FILE := ./benchmarks/bench.py
BENCH_MAT_MUL_FILE := ./benchmarks/bench_mat_mul.c

.PHONY: all clean format

all: xor full_adder iris lin_reg bench_mul

clear_net: $(LIB_FILE)

xor full_adder iris lin_reg: clear_net $(EXAMPLE_FILES)
	$(CC) $(CFLAGS) -o $@ $(EXAMPLE_DIR)/$@.c

run_%: %
	./$<

run_bench: iris full_adder xor
	python3 $(BENCH_FILE)

bench_mul: $(BENCH_MAT_MUL_FILE)
	$(CC) $(CFLAGS) -o $@ $<

run_bench_mul: clear_net bench_mul
	./$<

format:
	$(FORMAT) -style="{BasedOnStyle: llvm, IndentWidth: 4, TabWidth: 4, UseTab: Never}" -i $(EXAMPLE_FILES) $(LIB_FILE)

clean:
	rm -f xor full_adder iris lin_reg bench_mul
