CC := gcc
CFLAGS := -Wall -Wextra -pedantic
FORMAT := clang-format
LIB_FILE := clear_net.h
EXAMPLE_DIR := examples
EXAMPLE_FILES := $(wildcard $(EXAMPLE_DIR)/*.c)
C_FILES := $(shell find . -name '*.c' -not -path "./env/*")
H_FILES := $(shell find . -name '*.h' -not -path "./env/*")
BENCH_LEARN_FILE := ./benchmarks/bench_learn.py
BENCH_MAT_MUL_FILE := ./benchmarks/bench_mat_mul.c

examples := mnist xor full_adder iris lin_reg
clear_net: $(LIB_FILE)

$(examples): clear_net $(EXAMPLE_FILES)
	$(CC) $(CFLAGS) -o $@ $(EXAMPLE_DIR)/$@.c

bench_learn: iris $(BENCH_LEARN_FILE)
	python3 $(BENCH_LEARN_FILE)

bench_mul: clear_net $(BENCH_MAT_MUL_FILE)
	$(CC) $(CFLAGS) -o $@ $(BENCH_MAT_MUL_FILE)
	./$@

format: $(C_FILES) $(H_FILES)
	$(FORMAT) -style="{BasedOnStyle: llvm, IndentWidth: 4, TabWidth: 4, UseTab: Never}" -i $(C_FILES) $(H_FILES)

clean:
	rm -f $(examples) bench_mul
	rm -f model lin_reg_model adder_model
