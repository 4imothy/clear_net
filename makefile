CC := gcc
CFLAGS := -Wall -Wextra -pedantic
PY := python3

C_FORMAT := clang-format
PY_FORMAT := autopep8

LIB_DIR := lib
EXAMPLE_DIR := examples
SCRIPTS_DIR := scripts
BENCHMARK_DIR := $(SCRIPTS_DIR)/benchmarks
TEST_DIR := tests
DOWNLOAD_DATA_DIR := $(SCRIPTS_DIR)/download_datasets

LIB_C_FILES := $(wildcard $(LIB_DIR)/*.c)
LIB_H_FILES := $(wildcard $(LIB_DIR)/*.h)
EXAMPLE_FILES := $(wildcard $(EXAMPLE_DIR)/*.c)
TEST_FILES := $(wildcard $(TEST_DIR)/*.c)
TEST_HEADER := $(TEST_DIR)/tests.h
DOWNLOAD_SCRIPTS := $(wildcard $(DOWNLOAD_DATA_DIR)/*.py)

C_FILES := $(shell find . -name '*.c' -not -path "./env/*")
H_FILES := $(shell find . -name '*.h' -not -path "./env/*")
PY_FILES := $(shell find . -name '*.py' -not -path "./env/*")
BENCH_LEARN_FILE := $(BENCHMARK_DIR)/bench_learn.py
BENCH_MAT_MUL_FILE := $(BENCHMARK_DIR)/bench_mat_mul.c

examples := mnist_vanilla mnist_conv mnist_mix xor full_adder iris lin_reg mnist_conv_tiny
downloaders := get_mnist

.DEFAULT_GOAL := all

all: $(examples)

$(examples): $(LIB_C_FILES) $(LIB_H_FILES) $(EXAMPLE_FILES)
	$(CC) $(CFLAGS) -o $@ $(LIB_C_FILES) $(EXAMPLE_DIR)/$@.c

run_%: %
	./$<

tests := autodiff correlation forward_dense

$(tests): $(LIB_C_FILES) $(LIB_H_FILES) $(TEST_FILES) $(TEST_HEADER)
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/$@.c $(LIB_C_FILES)

test_%: %
	$(PY) $(TEST_DIR)/$<.py

test: $(addprefix test_, $(tests))

pip_update:
	pip --disable-pip-version-check list --outdated --format=json | python -c "import json, sys; print('\n'.join([x['name'] for x in json.load(sys.stdin)]))" | xargs -n1 pip install -U
	pip freeze > requirements.txt

$(downloaders): $(DOWNLOAD_SCRIPTS)
	$(PY) $(DOWNLOAD_DATA_DIR)/$@.py

bench_learn: iris $(BENCH_LEARN_FILE)
	$(PY) $(BENCH_LEARN_FILE)

format: $(C_FILES) $(H_FILES) $(PY_FILES)
	$(C_FORMAT) -style="{BasedOnStyle: llvm, IndentWidth: 4, TabWidth: 4, UseTab: Never}" -i $(C_FILES) $(H_FILES)
	$(PY_FORMAT) --in-place $(PY_FILES)

clean:
	rm -f $(examples) $(tests)
	rm -f model
