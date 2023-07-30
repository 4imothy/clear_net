#define CLEAR_NET_IMPLEMENTATION
#include "../../clear_net_new.h"
#include <stdio.h>
#include <string.h>

#define PRINT_VAL(x)                                                           \
    printf("%s %f %f\n", #x, GET_NODE((x)).num, GET_NODE((x)).grad)

int main(int argc, char *argv[]) {
    CLEAR_NET_ASSERT(argc == 2);
    NodeStore node_store = alloc_node_store(0);
    NodeStore *ns = &node_store;
    if (strcmp(argv[1], "1") == 0) {
        size_t a = init_leaf_var(ns, -2.0);
        size_t b = init_leaf_var(ns, 3.0);
        size_t c = multiply(ns, a, b);
        size_t d = add(ns, a, b);
        size_t e = multiply(ns, c, d);
        size_t f = subtract(ns, a, e);
        size_t g = hyper_tanv(ns, f);
        backward(ns, g);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
        PRINT_VAL(e);
        PRINT_VAL(f);
        PRINT_VAL(g);
    } else if (strcmp(argv[1], "2") == 0) {
        size_t one = init_leaf_var(ns, 1.0);
        size_t none = init_leaf_var(ns, -1.0);
        size_t two = init_leaf_var(ns, 2.0);
        size_t three = init_leaf_var(ns, 3.0);
        size_t a = init_leaf_var(ns, -4.0);
        size_t b = init_leaf_var(ns, 2.0);
        size_t c = add(ns, a, b);
        size_t d = add(ns, multiply(ns, a, b), b);
        c = add(ns, c, add(ns, c, one));
        c = add(ns, c, add(ns, c, add(ns, one, multiply(ns, none, a))));
        d = add(ns, d, add(ns, multiply(ns, d, two), reluv(ns, add(ns, b, a))));
        d = add(ns, d,
                add(ns, multiply(ns, d, three), reluv(ns, subtract(ns, b, a))));
        size_t e = subtract(ns, c, d);
        size_t f = reluv(ns, e);
        backward(ns, f);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
        PRINT_VAL(e);
        PRINT_VAL(f);
    } else if (strcmp(argv[1], "pow") == 0) {
        size_t a = init_leaf_var(ns, 5);
        size_t b = init_leaf_var(ns, 10);
        size_t c = raise(ns, a, b);
        c = raise(ns, c, init_leaf_var(ns, 2));
        backward(ns, c);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
    } else if (strcmp(argv[1], "on_itself") == 0) {
        size_t a = init_leaf_var(ns, 3.0);
        size_t b = init_leaf_var(ns, 7.0);
        size_t c = add(ns, a, b);
        size_t t = init_leaf_var(ns, 2.0);
        c = add(ns, c, t);
        c = add(ns, c, t);
        c = multiply(ns, c, a);
        c = subtract(ns, c, b);
        size_t d = init_leaf_var(ns, 5.0);
        d = subtract(ns, d, c);
        d = add(ns, d, d);
        backward(ns, d);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
    } else if (strcmp(argv[1], "tanh") == 0) {
        size_t x1 = init_leaf_var(ns, 2.0);
        size_t x2 = init_leaf_var(ns, -0.0);
        size_t w1 = init_leaf_var(ns, -3.0);
        size_t w2 = init_leaf_var(ns, 1.0);
        size_t b = init_leaf_var(ns, 7.0);
        size_t t1 = multiply(ns, x1, w1);
        size_t t2 = multiply(ns, x2, w2);
        size_t t3 = add(ns, t1, t2);
        size_t n = add(ns, t3, b);
        size_t o = hyper_tanv(ns, n);
        backward(ns, o);
        PRINT_VAL(x1);
        PRINT_VAL(w1);
        PRINT_VAL(x2);
        PRINT_VAL(w2);
    } else if (strcmp(argv[1], "relu") == 0) {
        size_t a = init_leaf_var(ns, 10.0);
        size_t b = init_leaf_var(ns, 5.0);
        size_t c = multiply(ns, a, b);
        size_t d = reluv(ns, c);
        backward(ns, d);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
    } else if (strcmp(argv[1], "sigmoid") == 0) {
        size_t a = init_leaf_var(ns, 0.3);
        size_t b = init_leaf_var(ns, 0.5);
        size_t c = init_leaf_var(ns, -1);
        size_t d = multiply(ns, c, add(ns, a, b));
        size_t e = multiply(ns, d, a);
        size_t f = sigmoidv(ns, e);
        backward(ns, f);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
        PRINT_VAL(e);
        PRINT_VAL(f);
    }
    dealloc_node_store(ns);
}
