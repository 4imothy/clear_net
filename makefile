CC := gcc
CFLAGS := -Wall -Wextra -pedantic
FORMAT := clang-format
LIB_FILE := clear_net.h
EXAMPLE_DIR := examples
EXAMPLE_FILES := $(shell find $(EXAMPLE_DIR) -name '*.c')

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

FORMAT_STYLE = -style="{BasedOnStyle: llvm, IndentWidth: 4, TabWidth: 4, UseTab: Never}"
# FORMAT_STYLE = -style"{IndentWidth: 4}"
format:
	$(FORMAT) $(FORMAT_STYLE) -i $(EXAMPLE_FILES) $(LIB_FILE)

clean:
	rm example
