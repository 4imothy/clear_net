#include "../lib/clear_net.h"
// function to learn
// y = 2 + 4a - 3b + 5c + 6d - 2e + 7f - 8g + 9h

#define data cn.data

scalar rand_range(scalar lower, scalar upper) {
    return ((scalar)rand()) / RAND_MAX * (upper - lower) + lower;
}

scalar do_func(scalar a,scalar b,scalar c,scalar d,scalar e,scalar f,scalar g,scalar h) {
    return 2 + (4 * a) -(3 * b) + (5 * c) + (6 * d) - (2 * e) + (7 * f) - (8 * g) + (9 * h);
}

int main(void) {
    srand(0);
    ulong num_train = 100;
    scalar lower = -1;
    scalar upper = 1;

    ulong dim_input = 8;
    ulong dim_output = 1;

    Vector *inputs = data.allocVectors(num_train, dim_input);
    Vector *val_inputs = data.allocVectors(num_train, dim_input);
    Vector *targets = data.allocVectors(num_train, dim_output);
    Vector *val_targets = data.allocVectors(num_train, dim_output);
    scalar max = do_func(upper, upper, upper, upper, upper, upper, upper, upper);

    scalar a;
    scalar b;
    scalar c;
    scalar d;
    scalar e;
    scalar f;
    scalar g;
    scalar h;
    for (ulong i = 0; i < num_train; ++i) {
        for (ulong j = 0; j < dim_input; ++j) {
            VEC_AT(inputs[i], j) = rand_range(lower, upper);
            VEC_AT(val_inputs[i], j) = rand_range(lower, upper);
        }
    }
    for (ulong i = 0; i < num_train; ++i) {
        a = VEC_AT(inputs[i], 0);
        b = VEC_AT(inputs[i], 1);
        c = VEC_AT(inputs[i], 2);
        d = VEC_AT(inputs[i], 3);
        e = VEC_AT(inputs[i], 4);
        f = VEC_AT(inputs[i], 5);
        g = VEC_AT(inputs[i], 6);
        h = VEC_AT(inputs[i], 7);
        VEC_AT(targets[i], 0) = do_func(a, b, c, d, e, f, g, h);
        VEC_AT(targets[i], 0) /= max;
        a = VEC_AT(val_inputs[i], 0);
        b = VEC_AT(val_inputs[i], 1);
        c = VEC_AT(val_inputs[i], 2);
        d = VEC_AT(val_inputs[i], 3);
        e = VEC_AT(val_inputs[i], 4);
        f = VEC_AT(val_inputs[i], 5);
        g = VEC_AT(val_inputs[i], 6);
        h = VEC_AT(val_inputs[i], 7);
        VEC_AT(val_targets[i], 0) = do_func(a, b, c, d, e, f, g, h);
        VEC_AT(val_targets[i], 0) /= max;
    }

    CNData *io_ins = data.allocDataFromVectors(inputs, num_train);
    CNData *io_tars = data.allocDataFromVectors(targets, num_train);
    CNData *io_val_ins = data.allocDataFromVectors(val_inputs, num_train);
    CNData *io_val_tars = data.allocDataFromVectors(val_targets, num_train);

    HParams *hp = cn.allocDefaultHParams();
    cn.setRate(hp, 0.01);
    Net *net = cn.allocVanillaNet(hp, 8);
    cn.allocDenseLayer(net, TANH, 1);
    cn.randomizeNet(net, -1, 1);
    ulong num_epochs = 200000;
    scalar error_break = 0.01f;
    scalar loss;
    for (ulong i = 0; i < num_epochs; ++i) {
        loss = cn.lossVanilla(net, io_ins, io_tars);
        cn.backprop(net);
        if (i % (num_epochs / 20) == 0) {
            printf("Cost at %zu: %f\n", i, loss);
        }
        if (loss < error_break) {
            printf("Less than: %f error at epoch %zu\n", error_break, i);
            break;
        }
    }

    printf("Final output: %f\n", cn.lossVanilla(net, io_ins, io_tars));
    cn.printVanillaPredictions(net, io_val_ins, io_val_tars);
    char *file_name = "model";
    cn.saveNet(net, file_name);
    cn.deallocNet(net);
    net = cn.allocNetFromFile(file_name);
    printf("After Loading From File\n");
    cn.printVanillaPredictions(net, io_val_ins, io_val_tars);
    cn.deallocNet(net);
    return 0;
}
