#include "../lib/clear_net.h"
// function to learn
// y = 2 + 4a - 3b + 5c + 6d - 2e + 7f - 8g + 9h

#define la cn.la

const ulong num_train = 100;
const ulong num_var = 8 + 1;
float train[num_train * num_var] = {0};
float val[num_train * num_var] = {0};

float rand_rangef(float lower, float upper) {
    return ((float)rand()) / RAND_MAX * (upper - lower) + lower;
}

#define do_func(a, b, c, d, e, f, g, h)                                        \
    (2 + 4 * (a)-3 * (b) + 5 * (c) + 6 * (d)-2 * (e) + 7 * (f)-8 * (g) +       \
     9 * (h))

const float len = 1;
const float lower = -1.0f * len;
const float upper = len;

// data is normalized
const float max = do_func(1, 1, 1, 1, 1, 1, 1, 1);

const ulong dim_input = num_var - 1;

float mul_max(float x) { return (x * max); }

int main(void) {
    srand(0);
    float a;
    float b;
    float c;
    float d;
    float e;
    float f;
    float g;
    float h;
    for (ulong i = 0; i < num_train; ++i) {
        for (ulong j = 0; j < num_var; ++j) {
            train[i * num_var + j] = rand_rangef(lower, upper);
            val[i * num_var + j] = rand_rangef(lower, upper);
            if (j == num_var - 1) {
                a = train[i * num_var];
                b = train[i * num_var + 1];
                c = train[i * num_var + 2];
                d = train[i * num_var + 3];
                e = train[i * num_var + 4];
                f = train[i * num_var + 5];
                g = train[i * num_var + 6];
                h = train[i * num_var + 7];
                train[i * num_var + num_var - 1] =
                    do_func(a, b, c, d, e, f, g, h) / max;

                a = val[i * num_var];
                b = val[i * num_var + 1];
                c = val[i * num_var + 2];
                d = val[i * num_var + 3];
                e = val[i * num_var + 4];
                f = val[i * num_var + 5];
                g = val[i * num_var + 6];
                h = val[i * num_var + 7];
                val[i * num_var + num_var - 1] =
                    do_func(a, b, c, d, e, f, g, h) / max;
            }
        }
    }

    Matrix input = la.formMatrix(num_train, dim_input, num_var, train);
    Matrix output = la.formMatrix(num_train, 1, num_var, &train[dim_input]);
    Matrix val_in = la.formMatrix(num_train, dim_input, num_var, val);
    Matrix val_out = la.formMatrix(num_train, 1, num_var, &val[dim_input]);

    HParams *hp = cn.allocDefaultHParams();
    cn.setRate(hp, 0.01);
    Net *net = cn.allocVanillaNet(hp, 8);
    cn.allocDenseLayer(net, Tanh, 1);
    cn.randomizeNet(net, -1, 1);
    ulong num_epochs = 200000;
    float error_break = 0.01f;
    float loss;
    for (ulong i = 0; i < num_epochs; ++i) {
        loss = cn.learnVanilla(net, input, output);
        if (i % (num_epochs / 20) == 0) {
            printf("Cost at %zu: %f\n", i, loss);
        }
        if (loss < error_break) {
            printf("Less than: %f error at epoch %zu\n", error_break, i);
            break;
        }
    }
    printf("Final output: %f\n", cn.lossVanilla(net, input, output));
    cn.printVanillaPredictions(net, val_in, val_out);
    char *file_name = "model";
    cn.saveNet(net, file_name);
    cn.deallocNet(net);
    net = cn.allocNetFromFile(file_name);
    printf("After Loading From File\n");
    cn.printVanillaPredictions(net, val_in, val_out);
    cn.deallocNet(net);
    return 0;
}
