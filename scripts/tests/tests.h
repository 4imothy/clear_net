#include <stdio.h>
#include <string.h>
#define CLEAR_NET_IMPLEMENTATION
#include "../../clear_net.h"

CLEAR_NET_DEFINE_HYPERPARAMETERS

int strequal(char *a, char *b) { return !(strcmp(a, b)); }
