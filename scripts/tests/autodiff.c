#define CLEAR_NET_IMPLEMENTATION
#include "../../clear_net_new.h"
#include <stdio.h>
#include <string.h>

#define PRINT_VAL(x) printf("%s %f %f\n", #x, GET_NODE((x)).num, GET_NODE((x)).grad)


int main(int argc, char *argv[]) {
    CLEAR_NET_ASSERT(argc == 2);
    NodeList node_list = alloc_node_list(0);
    NodeList *nl = &node_list;
    if (strcmp(argv[1], "1") == 0) {
        size_t a = init_leaf_var(nl, -2.0);
        size_t b = init_leaf_var(nl, 3.0);
        size_t c = multiply(nl, a, b);
        size_t d = add(nl, a, b);
        size_t e = multiply(nl, c, d);
        size_t f = subtract(nl, a, e);
        size_t g = hyper_tan(nl, f);
        backward(nl, g);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
        PRINT_VAL(e);
        PRINT_VAL(f);
        PRINT_VAL(g);
    } else if (strcmp(argv[1], "2") == 0) {
        size_t one = init_leaf_var(nl, 1.0);
        size_t none = init_leaf_var(nl, -1.0);
        size_t two = init_leaf_var(nl, 2.0);
        size_t three = init_leaf_var(nl, 3.0);
        size_t a = init_leaf_var(nl, -4.0);
        size_t b = init_leaf_var(nl, 2.0);
        size_t c = add(nl, a, b);
        size_t d = add(nl, multiply(nl, a, b), b);
        c = add(nl, c, add(nl, c, one));
        c = add(nl, c, add(nl, c, add(nl, one, multiply(nl, none, a))));
        d = add(nl, d, add(nl, multiply(nl, d, two), relu(nl, add(nl,b,a))));
        d = add(nl, d, add(nl, multiply(nl, d, three), relu(nl, subtract(nl, b, a))));
        size_t e = subtract(nl, c,d);
        size_t f = relu(nl, e);
        backward(nl, f);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
        PRINT_VAL(e);
        PRINT_VAL(f);
    }else if (strcmp(argv[1], "on_itself") == 0) {
        size_t a = init_leaf_var(nl, 3.0);
        size_t b = init_leaf_var(nl, 7.0);
        size_t c = add(nl, a, b);
        size_t t = init_leaf_var(nl, 2.0);
        c = add(nl, c, t);
        c = add(nl, c, t);
        c = multiply(nl, c, a);
        c = subtract(nl, c, b);
        size_t d = init_leaf_var(nl, 5.0);
        d = subtract(nl, d, c);
        d = add(nl, d, d);
        backward(nl, d);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
    } else if (strcmp(argv[1], "tanh") == 0) {
        size_t x1 = init_leaf_var(nl, 2.0);
        size_t x2 = init_leaf_var(nl, -0.0);
        size_t w1 = init_leaf_var(nl, -3.0);
        size_t w2 = init_leaf_var(nl, 1.0);
        size_t b = init_leaf_var(nl, 7.0);
        size_t t1 = multiply(nl, x1, w1);
        size_t t2 = multiply(nl, x2, w2);
        size_t t3 = add(nl, t1, t2);
        size_t n = add(nl, t3, b);
        size_t o = hyper_tan(nl, n);
        backward(nl, o);
        PRINT_VAL(x1);
        PRINT_VAL(w1);
        PRINT_VAL(x2);
        PRINT_VAL(w2);
    } else if (strcmp(argv[1], "relu") == 0) {
        size_t a = init_leaf_var(nl, 10.0);
        size_t b = init_leaf_var(nl, 5.0);
        size_t c = multiply(nl, a, b);
        size_t d = relu(nl, c);
        backward(nl, d);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
    } else if (strcmp(argv[1], "sigmoid") == 0) {
        size_t a = init_leaf_var(nl, 0.3);
        size_t b = init_leaf_var(nl, 0.5);
        size_t c = init_leaf_var(nl, -1);
        size_t d = multiply(nl, c, add(nl, a, b));
        size_t e = multiply(nl, d, a);
        size_t f = sigmoid(nl, e);
        backward(nl, f);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
        PRINT_VAL(e);
        PRINT_VAL(f);
    }
    deolloc_node_list(nl);
}
