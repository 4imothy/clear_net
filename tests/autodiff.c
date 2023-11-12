#include "../lib/clear_net.h"
#include "./tests.h"

#define ad cn.ad

#define PRINT_VAL(x)                                                           \
    printf("%s %f %f\n", #x, ad.getVal((cg), (x)), ad.getGrad((cg), (x)))

int main(int argc, char *argv[]) {
    CLEAR_NET_ASSERT(argc == 2);
    CompGraph *cg = ad.allocCompGraph(0);
    if (strequal(argv[1], "1")) {
        ulong a = ad.initLeafScalar(cg, -2.0);
        ulong b = ad.initLeafScalar(cg, 3.0);
        ulong c = ad.mul(cg, a, b);
        ulong d = ad.add(cg, a, b);
        ulong e = ad.mul(cg, c, d);
        ulong f = ad.sub(cg, a, e);
        ulong g = ad.htan(cg, f);
        ad.backprop(cg, g);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
        PRINT_VAL(e);
        PRINT_VAL(f);
        PRINT_VAL(g);
    } else if (strequal(argv[1], "2")) {
        ulong one = ad.initLeafScalar(cg, 1.0);
        ulong none = ad.initLeafScalar(cg, -1.0);
        ulong two = ad.initLeafScalar(cg, 2.0);
        ulong three = ad.initLeafScalar(cg, 3.0);
        ulong a = ad.initLeafScalar(cg, -4.0);
        ulong b = ad.initLeafScalar(cg, 2.0);
        ulong c = ad.add(cg, a, b);
        ulong d = ad.add(cg, ad.mul(cg, a, b), b);
        c = ad.add(cg, c, ad.add(cg, c, one));
        c = ad.add(cg, c, ad.add(cg, c, ad.add(cg, one, ad.mul(cg, none, a))));
        d = ad.add(cg, d, ad.add(cg, ad.mul(cg, d, two), ad.relu(cg, ad.add(cg, b, a))));
        d = ad.add(cg, d, ad.add(cg, ad.mul(cg, d, three), ad.relu(cg, ad.sub(cg, b, a))));
        ulong e = ad.sub(cg, c, d);
        ulong f = ad.relu(cg, e);
        ad.backprop(cg, f);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
        PRINT_VAL(e);
        PRINT_VAL(f);
    } else if (strequal(argv[1], "pow")) {
        ulong a = ad.initLeafScalar(cg, 5);
        ulong b = ad.initLeafScalar(cg, 10);
        ulong c = ad.raise(cg, a, b);
        c = ad.raise(cg, c, ad.initLeafScalar(cg, 2));
        ad.backprop(cg, c);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
    } else if (strequal(argv[1], "on_itself")) {
        ulong a = ad.initLeafScalar(cg, 3.0);
        ulong b = ad.initLeafScalar(cg, 7.0);
        ulong c = ad.add(cg, a, b);
        ulong t = ad.initLeafScalar(cg, 2.0);
        c = ad.add(cg, c, t);
        c = ad.add(cg, c, t);
        c = ad.mul(cg, c, a);
        c = ad.sub(cg, c, b);
        ulong d = ad.initLeafScalar(cg, 5.0);
        d = ad.sub(cg, d, c);
        d = ad.add(cg, d, d);
        ad.backprop(cg, d);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
    } else if (strequal(argv[1], "tanh")) {
        ulong x1 = ad.initLeafScalar(cg, 2.0);
        ulong x2 = ad.initLeafScalar(cg, -0.0);
        ulong w1 = ad.initLeafScalar(cg, -3.0);
        ulong w2 = ad.initLeafScalar(cg, 1.0);
        ulong b = ad.initLeafScalar(cg, 7.0);
        ulong t1 = ad.mul(cg, x1, w1);
        ulong t2 = ad.mul(cg, x2, w2);
        ulong t3 = ad.add(cg, t1, t2);
        ulong n = ad.add(cg, t3, b);
        ulong o = ad.htan(cg, n);
        ad.backprop(cg, o);
        PRINT_VAL(x1);
        PRINT_VAL(w1);
        PRINT_VAL(x2);
        PRINT_VAL(w2);
    } else if (strequal(argv[1], "relu")) {
        ulong a = ad.initLeafScalar(cg, 10.0);
        ulong b = ad.initLeafScalar(cg, 5.0);
        ulong c = ad.mul(cg, a, b);
        ulong d = ad.relu(cg, c);
        ad.backprop(cg, d);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
    } else if (strequal(argv[1], "sigmoid")) {
        ulong a = ad.initLeafScalar(cg, 0.3);
        ulong b = ad.initLeafScalar(cg, 0.5);
        ulong c = ad.initLeafScalar(cg, -1);
        ulong d = ad.mul(cg, c, ad.add(cg, a, b));
        ulong e = ad.mul(cg, d, a);
        ulong f = ad.sigmoid(cg, e);
        ad.backprop(cg, f);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
        PRINT_VAL(e);
        PRINT_VAL(f);
    } else if (strequal(argv[1], "leaky_relu")) {
        ulong a = ad.initLeafScalar(cg, 72);
        ulong b = ad.initLeafScalar(cg, 38);
        ulong c = ad.initLeafScalar(cg, -10);
        ulong d = ad.mul(cg, c, ad.add(cg, a, b));
        ulong e = ad.mul(cg, d, a);
        ulong f = ad.leakyRelu(cg, e);
        ad.backprop(cg, f);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
        PRINT_VAL(e);
        PRINT_VAL(f);
    } else if (strequal(argv[1], "elu")) {
        ulong a = ad.initLeafScalar(cg, 5);
        ulong b = ad.initLeafScalar(cg, -6);
        ulong c = ad.elu(cg, b);
        ulong d = ad.mul(cg, a, ad.sub(cg, c, b));
        ulong e = ad.mul(cg, d, d);
        ulong f = ad.elu(cg, e);
        ad.backprop(cg, f);
        PRINT_VAL(a);
        PRINT_VAL(b);
        PRINT_VAL(c);
        PRINT_VAL(d);
        PRINT_VAL(e);
        PRINT_VAL(f);
    }

    ad.deallocCompGraph(cg);
}
