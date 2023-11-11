CC := gcc
CFLAGS := -Wall -Wextra -pedantic
PY := python3

C_FORMAT := clang-format
PY_FORMAT := autopep8

LIB_DIR := clear_net
EXAMPLE_DIR := examples
SCRIPTS_DIR := scripts
BENCHMARK_DIR := $(SCRIPTS_DIR)/benchmarks
TEST_DIR := $(SCRIPTS_DIR)/tests
DOWNLOAD_DATA_DIR := $(SCRIPTS_DIR)/download_datasets

LIB_FILES := $(wildcard $(LIB_DIR)/*.c)
EXAMPLE_FILES := $(wildcard $(EXAMPLE_DIR)/*.c)
TEST_FILES := $(wildcard $(TEST_DIR)/*.c)
DOWNLOAD_SCRIPTS := $(wildcard $(DOWNLOAD_DATA_DIR)/*.py)

C_FILES := $(shell find . -name '*.c' -not -path "./env/*")
H_FILES := $(shell find . -name '*.h' -not -path "./env/*")
PY_FILES := $(shell find . -name '*.py' -not -path "./env/*")
BENCH_LEARN_FILE := $(BENCHMARK_DIR)/bench_learn.py
BENCH_MAT_MUL_FILE := $(BENCHMARK_DIR)/bench_mat_mul.c

examples := mnist_vanilla mnist_conv mnist_mix xor full_adder iris lin_reg mnist_conv_tiny
downloaders := get_mnist

clear_net: $(LIB_FILES)

.DEFAULT_GOAL := all

all: $(examples)

$(examples): clear_net $(EXAMPLE_FILES)
	$(CC) $(CFLAGS) -o $@ $(EXAMPLE_DIR)/$@.c

run_%: %
	./$<

tests := autodiff correlation gs_forward

$(tests): clear_net $(TEST_FILES)
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/$@.c $(LIB_FILES)

test_%: %
	$(PY) $(TEST_DIR)/$<.py

test: $(addprefix test_, $(tests))

run_test: $(LIB_FILES) test.c
	$(CC) $(CFLAGS) -o $@ test.c $(LIB_FILES)
	./run_test

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
