#include "./tests.h"
#include "../../clear_net/defines.h"
#include "../../clear_net/autodiff.h"

#define PRINT_VAL(x)                                                           \
    printf("%s %f %f\n", #x, cg->vars[(x)].num, cg->vars[(x)].grad)

int main(int argc, char *argv[]) {
    CLEAR_NET_ASSERT(argc == 2);
    CompGraph comp_graph = allocCompGraph(0);
    CompGraph *cg = &comp_graph;
    if (strequal(argv[1], "1")) {
        ulong a = initLeafScalar(cg, -2.0);
        ulong b = initLeafScalar(cg, 3.0);
        ulong c = mul(cg, a, b);
        ulong d = add(cg, a, b);
        ulong e = mul(cg, c, d);
        ulong f = sub(cg, a, e);
        ulong g = htan(cg, f);
        backprop(cg, g);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
        PRINT_VAL(e);
        PRINT_VAL(f);
        PRINT_VAL(g);
    } else if (strequal(argv[1], "2")) {
        ulong one = initLeafScalar(cg, 1.0);
        ulong none = initLeafScalar(cg, -1.0);
        ulong two = initLeafScalar(cg, 2.0);
        ulong three = initLeafScalar(cg, 3.0);
        ulong a = initLeafScalar(cg, -4.0);
        ulong b = initLeafScalar(cg, 2.0);
        ulong c = add(cg, a, b);
        ulong d = add(cg, mul(cg, a, b), b);
        c = add(cg, c, add(cg, c, one));
        c = add(cg, c,
                   add(cg, c, add(cg, one, mul(cg, none, a))));
        d = add(cg, d,
                   add(cg, mul(cg, d, two),
                          relu(cg, add(cg, b, a))));
        d = add(cg, d,
                   add(cg, mul(cg, d, three),
                          relu(cg, sub(cg, b, a))));
        ulong e = sub(cg, c, d);
        ulong f = relu(cg, e);
        backprop(cg, f);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
        PRINT_VAL(e);
        PRINT_VAL(f);
    } else if (strequal(argv[1], "pow")) {
        ulong a = initLeafScalar(cg, 5);
        ulong b = initLeafScalar(cg, 10);
        ulong c = raise(cg, a, b);
        c = raise(cg, c, initLeafScalar(cg, 2));
        backprop(cg, c);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
    } else if (strequal(argv[1], "on_itself")) {
        ulong a = initLeafScalar(cg, 3.0);
        ulong b = initLeafScalar(cg, 7.0);
        ulong c = add(cg, a, b);
        ulong t = initLeafScalar(cg, 2.0);
        c = add(cg, c, t);
        c = add(cg, c, t);
        c = mul(cg, c, a);
        c = sub(cg, c, b);
        ulong d = initLeafScalar(cg, 5.0);
        d = sub(cg, d, c);
        d = add(cg, d, d);
        backprop(cg, d);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
    } else if (strequal(argv[1], "tanh")) {
        ulong x1 = initLeafScalar(cg, 2.0);
        ulong x2 = initLeafScalar(cg, -0.0);
        ulong w1 = initLeafScalar(cg, -3.0);
        ulong w2 = initLeafScalar(cg, 1.0);
        ulong b = initLeafScalar(cg, 7.0);
        ulong t1 = mul(cg, x1, w1);
        ulong t2 = mul(cg, x2, w2);
        ulong t3 = add(cg, t1, t2);
        ulong n = add(cg, t3, b);
        ulong o = htan(cg, n);
        backprop(cg, o);
        PRINT_VAL(x1);
        PRINT_VAL(w1);
        PRINT_VAL(x2);
        PRINT_VAL(w2);
    } else if (strequal(argv[1], "relu")) {
        ulong a = initLeafScalar(cg, 10.0);
        ulong b = initLeafScalar(cg, 5.0);
        ulong c = mul(cg, a, b);
        ulong d = relu(cg, c);
        backprop(cg, d);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
    } else if (strequal(argv[1], "sigmoid")) {
        ulong a = initLeafScalar(cg, 0.3);
        ulong b = initLeafScalar(cg, 0.5);
        ulong c = initLeafScalar(cg, -1);
        ulong d = mul(cg, c, add(cg, a, b));
        ulong e = mul(cg, d, a);
        ulong f = sigmoid(cg, e);
        backprop(cg, f);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
        PRINT_VAL(e);
        PRINT_VAL(f);
    } else if (strequal(argv[1], "leaky_relu")) {
        ulong a = initLeafScalar(cg, 72);
        ulong b = initLeafScalar(cg, 38);
        ulong c = initLeafScalar(cg, -10);
        ulong d = mul(cg, c, add(cg, a, b));
        ulong e = mul(cg, d, a);
        ulong f = leakyRelu(cg, e);
        backprop(cg, f);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
        PRINT_VAL(e);
        PRINT_VAL(f);
    } else if (strequal(argv[1], "elu")) {
        ulong a = initLeafScalar(cg, 5);
        ulong b = initLeafScalar(cg, -6);
        ulong c = elu(cg, b);
        ulong d = mul(cg, a, sub(cg, c, b));
        ulong e = mul(cg, d, d);
        ulong f = elu(cg, e);
        backprop(cg, f);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
        PRINT_VAL(e);
        PRINT_VAL(f);
    }

    deallocCompGraph(cg);
}
