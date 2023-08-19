#define CLEAR_NET_IMPLEMENTATION
#include "../../clear_net.h"
#include <stdio.h>
#include <string.h>

#define PRINT_VAL(x)                                                           \
    printf("%s %f %f\n", #x, GET_NODE((x)).num, GET_NODE((x)).grad)

int main(int argc, char *argv[]) {
    cn_default_hparams();
    CLEAR_NET_ASSERT(argc == 2);
    GradientStore gradient_store = cn_alloc_gradient_store(0);
    GradientStore *gs = &gradient_store;
    if (strcmp(argv[1], "1") == 0) {
        size_t a = cn_init_leaf_var(gs, -2.0);
        size_t b = cn_init_leaf_var(gs, 3.0);
        size_t c = cn_multiply(gs, a, b);
        size_t d = cn_add(gs, a, b);
        size_t e = cn_multiply(gs, c, d);
        size_t f = cn_subtract(gs, a, e);
        size_t g = cn_hyper_tanv(gs, f);
        cn_backward(gs, g);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
        PRINT_VAL(e);
        PRINT_VAL(f);
        PRINT_VAL(g);
    } else if (strcmp(argv[1], "2") == 0) {
        size_t one = cn_init_leaf_var(gs, 1.0);
        size_t none = cn_init_leaf_var(gs, -1.0);
        size_t two = cn_init_leaf_var(gs, 2.0);
        size_t three = cn_init_leaf_var(gs, 3.0);
        size_t a = cn_init_leaf_var(gs, -4.0);
        size_t b = cn_init_leaf_var(gs, 2.0);
        size_t c = cn_add(gs, a, b);
        size_t d = cn_add(gs, cn_multiply(gs, a, b), b);
        c = cn_add(gs, c, cn_add(gs, c, one));
        c = cn_add(gs, c,
                   cn_add(gs, c, cn_add(gs, one, cn_multiply(gs, none, a))));
        d = cn_add(gs, d,
                   cn_add(gs, cn_multiply(gs, d, two),
                          cn_reluv(gs, cn_add(gs, b, a))));
        d = cn_add(gs, d,
                   cn_add(gs, cn_multiply(gs, d, three),
                          cn_reluv(gs, cn_subtract(gs, b, a))));
        size_t e = cn_subtract(gs, c, d);
        size_t f = cn_reluv(gs, e);
        cn_backward(gs, f);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
        PRINT_VAL(e);
        PRINT_VAL(f);
    } else if (strcmp(argv[1], "pow") == 0) {
        size_t a = cn_init_leaf_var(gs, 5);
        size_t b = cn_init_leaf_var(gs, 10);
        size_t c = cn_raise(gs, a, b);
        c = cn_raise(gs, c, cn_init_leaf_var(gs, 2));
        cn_backward(gs, c);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
    } else if (strcmp(argv[1], "on_itself") == 0) {
        size_t a = cn_init_leaf_var(gs, 3.0);
        size_t b = cn_init_leaf_var(gs, 7.0);
        size_t c = cn_add(gs, a, b);
        size_t t = cn_init_leaf_var(gs, 2.0);
        c = cn_add(gs, c, t);
        c = cn_add(gs, c, t);
        c = cn_multiply(gs, c, a);
        c = cn_subtract(gs, c, b);
        size_t d = cn_init_leaf_var(gs, 5.0);
        d = cn_subtract(gs, d, c);
        d = cn_add(gs, d, d);
        cn_backward(gs, d);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
    } else if (strcmp(argv[1], "tanh") == 0) {
        size_t x1 = cn_init_leaf_var(gs, 2.0);
        size_t x2 = cn_init_leaf_var(gs, -0.0);
        size_t w1 = cn_init_leaf_var(gs, -3.0);
        size_t w2 = cn_init_leaf_var(gs, 1.0);
        size_t b = cn_init_leaf_var(gs, 7.0);
        size_t t1 = cn_multiply(gs, x1, w1);
        size_t t2 = cn_multiply(gs, x2, w2);
        size_t t3 = cn_add(gs, t1, t2);
        size_t n = cn_add(gs, t3, b);
        size_t o = cn_hyper_tanv(gs, n);
        cn_backward(gs, o);
        PRINT_VAL(x1);
        PRINT_VAL(w1);
        PRINT_VAL(x2);
        PRINT_VAL(w2);
    } else if (strcmp(argv[1], "relu") == 0) {
        size_t a = cn_init_leaf_var(gs, 10.0);
        size_t b = cn_init_leaf_var(gs, 5.0);
        size_t c = cn_multiply(gs, a, b);
        size_t d = cn_reluv(gs, c);
        cn_backward(gs, d);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
    } else if (strcmp(argv[1], "sigmoid") == 0) {
        size_t a = cn_init_leaf_var(gs, 0.3);
        size_t b = cn_init_leaf_var(gs, 0.5);
        size_t c = cn_init_leaf_var(gs, -1);
        size_t d = cn_multiply(gs, c, cn_add(gs, a, b));
        size_t e = cn_multiply(gs, d, a);
        size_t f = cn_sigmoidv(gs, e);
        cn_backward(gs, f);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
        PRINT_VAL(e);
        PRINT_VAL(f);
    } else if (strcmp(argv[1], "leaky_relu") == 0) {
        size_t a = cn_init_leaf_var(gs, 72);
        size_t b = cn_init_leaf_var(gs, 38);
        size_t c = cn_init_leaf_var(gs, -10);
        size_t d = cn_multiply(gs, c, cn_add(gs, a, b));
        size_t e = cn_multiply(gs, d, a);
        size_t f = cn_leaky_reluv(gs, e);
        cn_backward(gs, f);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
        PRINT_VAL(e);
        PRINT_VAL(f);
    } else if (!strcmp(argv[1], "elu")) {
        // TODO make all the if statements like this
        size_t a = cn_init_leaf_var(gs, 5);
        size_t b = cn_init_leaf_var(gs, -6);
        size_t c = cn_eluv(gs, b);
        size_t d = cn_multiply(gs, a, cn_subtract(gs, c, b));
        size_t e = cn_multiply(gs, d, d);
        size_t f = cn_eluv(gs, e);
        cn_backward(gs, f);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
        PRINT_VAL(e);
        PRINT_VAL(f);
    }

    cn_dealloc_gradient_store(gs);
}
