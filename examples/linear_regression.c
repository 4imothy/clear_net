#define CLEAR_NET_IMPLEMENTATION
// #define CLEAR_NET_RATE 0.01f
#include "../clear_net.h"
// function to learn
// y = 2 + 4a - 3b + 5c + 6d - 2e + 7f - 8g + 9h

const size_t num_train = 100;
const size_t num_var = 8 + 1;
float train[num_train * num_var] = {0};
float val[num_train * num_var] = {0};

float rand_rangef(float lower, float upper) {
    return ((float)rand()) / RAND_MAX * (upper - lower) + lower;
}

#define do_func(a, b, c, d, e, f, g, h)                                        \
    (2 + 4 * (a)-3 * (b) + 5 * (c) + 6 * (d)-2 * (e) + 7 * (f)-8 * (g) +       \
     9 * (h))

const float lower = -100;
const float upper = 100;

const float max =
    do_func(upper, upper, upper, upper, upper, upper, upper, upper);

float mul_max(float x) { return (x * 2 * max) - max; }

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
    for (size_t i = 0; i < num_train; ++i) {
        for (size_t j = 0; j < num_var; ++j) {
            train[i * num_var + j] = rand_rangef(lower, upper);
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
                    (do_func(a, b, c, d, e, f, g, h) + max) / (max * 2);
            }
        }
    }
    for (size_t i = 0; i < num_train; ++i) {
        for (size_t j = 0; j < num_var; ++j) {
            val[i * num_var + j] = rand_rangef(lower, upper);
            if (j == num_var - 1) {
                a = val[i * num_var];
                b = val[i * num_var + 1];
                c = val[i * num_var + 2];
                d = val[i * num_var + 3];
                e = val[i * num_var + 4];
                f = val[i * num_var + 5];
                g = val[i * num_var + 6];
                h = val[i * num_var + 7];
                val[i * num_var + num_var - 1] =
                    (do_func(a, b, c, d, e, f, g, h) + max) / (max * 2);
            }
        }
    }

    size_t dim_input = num_var - 1;
    Matrix input = mat_form(num_train, dim_input, num_var, train);
    Matrix output = mat_form(num_train, 1, num_var, &train[dim_input]);
    Matrix val_in = mat_form(num_train, dim_input, num_var, val);
    Matrix val_out = mat_form(num_train, 1, num_var, &val[dim_input]);
    size_t  shape[] = {dim_input, 16, 5, 5, 1};
    Net net = alloc_net(shape, ARR_LEN(shape));
    net_rand(net, -1, 1);
    size_t num_epochs = 200000;
    float error_break = 0.001f;
    float error;
    for (size_t i = 0; i < num_epochs; ++i) {
        net_backprop(net, input, output);
        error = net_errorf(net, input, output);
        if (i % (num_epochs / 20) == 0) {
            printf("Cost at %zu: %f\n", i, error);
        }
        if (error < error_break) {
            printf("Less than: %f error at epoch %zu\n", error_break, i);
            break;
        }
    }
    NET_PRINT(net);
    /* net_print_results(net, input, output, &mul_max); */
    /* char *file_name = "lin_reg_model"; */
    /* net_save_to_file(file_name, net); */
    /* dealloc_net(&net); */
    /* net = alloc_net_from_file(file_name); */
    /* printf("After Loading\n"); */
    /* net_print_results(net, input, output, &mul_max); */
    /* net_print_results(net, val_in, val_out, &mul_max); */
    return 0;
}
