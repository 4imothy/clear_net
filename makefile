CC := gcc
CFLAGS := -Wall -Wextra -pedantic
FORMAT := clang-format
LIB_FILE := clear_net.h
EXAMPLE_DIR := examples
EXAMPLE_FILES := $(wildcard $(EXAMPLE_DIR)/*.c)
BENCH_FILE := ./benchmarks/bench.py
BENCH_MAT_MUL_FILE := ./benchmarks/bench_mat_mul.c
C_FILES := $(shell find . -name '*.c' -not -path "./env/*")
H_FILES := $(shell find . -name '*.h' -not -path "./env/*")

clear_net: $(LIB_FILE)

mnist xor full_adder iris lin_reg: clear_net $(EXAMPLE_FILES)
	$(CC) $(CFLAGS) -o $@ $(EXAMPLE_DIR)/$@.c

run_%: %
	./$<

# benching only implemented for iris examples
run_bench: iris
	python3 $(BENCH_FILE)

bench_mul: $(BENCH_MAT_MUL_FILE)
	$(CC) $(CFLAGS) -o $@ $<

run_bench_mul: bench_mul clear_net
	./$<

format: $(C_FILES) $(H_FILES)
	$(FORMAT) -style="{BasedOnStyle: llvm, IndentWidth: 4, TabWidth: 4, UseTab: Never}" -i $(C_FILES) $(H_FILES)

clean:
	rm -f xor full_adder iris lin_reg mnist bench_mul
	rm -f model lin_reg_model adder_model
