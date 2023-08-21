CC := gcc
CFLAGS := -Wall -Wextra -pedantic
PY := python3

C_FORMAT := clang-format
PY_FORMAT := autopep8

LIB_FILE := clear_net.h
EXAMPLE_DIR := examples
SCRIPTS_DIR := scripts
BENCHMARK_DIR := $(SCRIPTS_DIR)/benchmarks
TEST_DIR := $(SCRIPTS_DIR)/tests
DOWNLOAD_DATA_DIR := $(SCRIPTS_DIR)/download_datasets

EXAMPLE_FILES := $(wildcard $(EXAMPLE_DIR)/*.c)
TEST_FILES := $(wildcard $(TEST_DIR)/*.c)
DOWNLOAD_SCRIPTS := $(wildcard $(DOWNLOAD_DATA_DIR)/*.py)

C_FILES := $(shell find . -name '*.c' -not -path "./env/*")
H_FILES := $(shell find . -name '*.h' -not -path "./env/*")
PY_FILES := $(shell find . -name '*.py' -not -path "./env/*")
BENCH_LEARN_FILE := $(BENCHMARK_DIR)/bench_learn.py
BENCH_MAT_MUL_FILE := $(BENCHMARK_DIR)/bench_mat_mul.c

examples := mnist_vanilla mnist_conv mnist_mix xor full_adder iris lin_reg
downloaders := get_mnist
clear_net: $(LIB_FILE)

$(examples): clear_net $(EXAMPLE_FILES)
	$(CC) $(CFLAGS) -o $@ $(EXAMPLE_DIR)/$@.c

run_%: %
	./$<

tests := autodiff convolution

$(tests): clear_net $(TEST_FILES)
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/$@.c

test_%: %
	$(PY) $(TEST_DIR)/$<.py

test: $(addprefix test_, $(tests))

$(downloaders): $(DOWNLOAD_SCRIPTS)
	$(PY) $(DOWNLOAD_DATA_DIR)/$@.py

bench_learn: iris $(BENCH_LEARN_FILE)
	$(PY) $(BENCH_LEARN_FILE)

bench_mul: clear_net $(BENCH_MAT_MUL_FILE)
	$(CC) $(CFLAGS) -o $@ $(BENCH_MAT_MUL_FILE)
	./$@

format: $(C_FILES) $(H_FILES) $(PY_FILES)
	$(C_FORMAT) -style="{BasedOnStyle: llvm, IndentWidth: 4, TabWidth: 4, UseTab: Never}" -i $(C_FILES) $(H_FILES)
	$(PY_FORMAT) --in-place $(PY_FILES)

clean:
	rm -f $(examples) $(tests) bench_mul
	rm -f model lin_reg_model adder_model
