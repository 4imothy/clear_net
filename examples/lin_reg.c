#define CLEAR_NET_IMPLEMENTATION
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

const float len = 1;
const float lower = -1.0f * len;
const float upper = len;

// data is normalized
const float max = do_func(1, 1, 1, 1, 1, 1, 1, 1);

const size_t dim_input = num_var - 1;

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
    for (size_t i = 0; i < num_train; ++i) {
        for (size_t j = 0; j < num_var; ++j) {
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

    Matrix input = cn_form_matrix(num_train, dim_input, num_var, train);
    Matrix output = cn_form_matrix(num_train, 1, num_var, &train[dim_input]);
    Matrix val_in = cn_form_matrix(num_train, dim_input, num_var, val);
    Matrix val_out = cn_form_matrix(num_train, 1, num_var, &val[dim_input]);

    cn_default_hparams();
    cn_set_rate(0.01);
    Net net = cn_init_net();
    cn_alloc_dense_layer(&net, Tanh, 8, 1);
    cn_randomize_net(&net, -1, 1);
    size_t num_epochs = 200000;
    float error_break = 0.01f;
    float loss;
    for (size_t i = 0; i < num_epochs; ++i) {
        loss = cn_learn_vani(&net, input, output);
        if (i % (num_epochs / 20) == 0) {
            printf("Cost at %zu: %f\n", i, loss);
        }
        if (loss < error_break) {
            printf("Less than: %f error at epoch %zu\n", error_break, i);
            break;
        }
    }
    printf("Final output: %f\n", cn_loss_vani(&net, input, output));
    cn_print_vani_results(net, input, output);
    char *file_name = "model";
    cn_save_net_to_file(net, file_name);
    cn_dealloc_net(&net);
    net = cn_alloc_net_from_file(file_name);
    printf("After Loading From File\n");
    cn_print_vani_results(net, val_in, val_out);
    cn_dealloc_net(&net);
    return 0;
}
