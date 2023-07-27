/* License
   Clear Net by Timothy Cronin

   To the extent possible under law, the person who associated CC0 with
   Clear Net has waived all copyright and related or neighboring rights
   to Clear Net.

   You should have received a copy of the CC0 legalcode along with this
   work.  If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
*/
/* Clear Net
  A framework for the creation and training of arbitrarily sized neural nets.
  Features:
    - Training with gradient descent and backpropagation
    - Optionall can utilize gradient descent with momentum with these equations
        - v_{dw} = \beta v_{dw} + (1-\beta)dw
        - W = W - \alpha v_{dw}
        - v_{db} = \beta v_{db} + (1-\beta)db
        - b = b - \alpha v_{db}
    - Simple interface for hyperparameter tuning (see macros below)
    - Ability to save and load a neural net to a file
    - Customize the activation functions for output and hidden layers
    - Multiple activation functions: Sigmoid, ReLU, Leaky_ReLU, Tanh, ELU

  Basic Structure of a Net:
    Each layer consists of weights -> biases -> activation.
    In each forward, each layer takes the previous activation matrix and
  multiplies it with the weights, then adds the bias and then applies the
  correct activation function. Each column in the weight matrix is a neuron.
    Each activation * weights results in a single column matrix
    which the bias is added to.

  Below are the definitions of structs, enums, macros and the
  declaractions of functions that are defined later.
  Some functions are commented out to abstract and
  keep users' namespace sane.
*/

#ifndef CLEAR_NET
#define CLEAR_NET

#include <math.h>   // expf
#include <stdio.h>  // printf
#include <stdlib.h> // malloc, free, size_t

#define ARR_LEN(a) (sizeof((a)) / sizeof((*a)))

// allow custom memory allocation strategies
#ifndef CLEAR_NET_ALLOC
#define CLEAR_NET_ALLOC malloc
#endif // CLEAR_NET_ALLOC
#ifndef CLEAR_NET_REALLOC
#define CLEAR_NET_REALLOC realloc
#endif // CLEAR_NET_REALLOC
// allow custom memory free strategies
#ifndef CLEAR_NET_DEALLOC
#define CLEAR_NET_DEALLOC free
#endif // CLEAR_NET_MALLOC
// allow custom assertion strategies
#ifndef CLEAR_NET_ASSERT
#include "assert.h" // assert
#define CLEAR_NET_ASSERT assert
#endif // CLEAR_NET_ASSERT

#ifndef CLEAR_NET_RATE
#define CLEAR_NET_RATE 0.5f
#endif
#ifndef CLEAR_NET_ACT_OUTPUT
#define CLEAR_NET_ACT_OUTPUT Sigmoid
#endif // CLEAR_NET_ACT_OUTPUT
#ifndef CLEAR_NET_ACT_HIDDEN
#define CLEAR_NET_ACT_HIDDEN Leaky_ReLU
#endif // CLEAR_NET_ACT_HIDDEN
#ifndef CLEAR_NET_ACT_NEG_SCALE
#define CLEAR_NET_ACT_NEG_SCALE 0.1f
#endif // CLEAR_NET_NEG_SCALE

#ifndef CLEAR_NET_MOMENTUM
#define CLEAR_NET_MOMENTUM 0
#endif // CLEAR_NET_MOMENTUM
#ifndef CLEAR_NET_MOMENTUM_BETA
#define CLEAR_NET_MOMENTUM_BETA 0.9
#endif // CLEAR_NET_MOMENTUM_BETA

// float randf(void);

// float reluf(float x);
// float actf(float x, Activation act);
// float dactf(float y, Activation act);
typedef enum {
    Sigmoid,
    ReLU,
    Leaky_ReLU,
    Tanh,
    ELU,
} Activation;

typedef struct {
    // define the shape
    size_t nrows;
    size_t ncols;
    size_t stride;
    // pointer to first element
    float *elements;
} Matrix;

#define MAT_GET(mat, r, c) (mat).elements[(r) * (mat).stride + (c)]
#define MAT_PRINT(mat) mat_print(mat, #mat)

Matrix alloc_mat(size_t nrows, size_t ncols);
void dealloc_mat(Matrix *mat);
Matrix mat_form(size_t nrows, size_t ncols, size_t stride, float *elements);
void mat_print(Matrix mat, char *name);
Matrix mat_row(Matrix giver, size_t row);
void mat_copy(Matrix dest, Matrix giver);
void mat_randomize_rows(Matrix mat);
// void mat_write_to_file(FILE *f, Matrix mat);
// void mat_mul(Matrix dest, Matrix left, Matrix right);
// void mat_sum(Matrix dest, Matrix toAdd);
// void mat_rand(Matrix mat, float lower, float upper);
// void mat_act(Matrix mat);

typedef struct {
    size_t nlayers;
    Matrix *activations;
    // number of these is equal to the number of layers -1 (for the output)
    Matrix *weights;
    Matrix *biases;
    size_t *shape;
    // below are extra stores for backprop
    Matrix *weight_alters; // store differences in weights
    Matrix *act_alters;    // store differences in activations
    // these are only allocated if momentum is enabled
    Matrix *momentum_weight_store;
    Matrix *momentum_bias_store;
} Net;

#define NET_INPUT(net)                                                         \
    (CLEAR_NET_ASSERT((net).nlayers > 0), (net).activations[0])
#define NET_OUTPUT(net)                                                        \
    (CLEAR_NET_ASSERT((net).nlayers > 0), (net).activations[(net).nlayers - 1])
#define NET_PRINT(net) net_print(net, #net)

Net alloc_net(size_t *shape, size_t nlayers);
void dealloc_net(Net *net, size_t shape_allocated);
Net alloc_net_from_file(char *file_name);
void net_save_to_file(char *file_name, Net net);
float net_errorf(Net net, Matrix input, Matrix target);
void net_print(Net net, char *name);
void net_rand(Net net, float low, float high);
void net_backprop(Net net, Matrix input, Matrix target);
void net_print_results(Net net, Matrix input, Matrix target,
                       float (*fix_output)(float));
void net_get_batch(Matrix *batch_input, Matrix *batch_output, Matrix input,
                   Matrix output, size_t batch_num, size_t batch_size);
// void net_forward(Net net);

#endif // CLEAR_NET

#ifdef CLEAR_NET_IMPLEMENTATION

/************************/
/* Activation Functions */
/************************/
float actf(float x, Activation act) {
    switch (act) {
    case Sigmoid:
        return 1 / (1 + expf(-x));
    case ReLU:
        return x > 0 ? x : 0;
    case Leaky_ReLU:
        return x >= 0 ? x : CLEAR_NET_ACT_NEG_SCALE * x;
    case Tanh: {
        float e_to_x = expf(x);
        float e_to_neg_x = expf(-x);
        return (e_to_x - e_to_neg_x) / (e_to_x + e_to_neg_x);
    }
    case ELU:
        return x > 0 ? x : CLEAR_NET_ACT_NEG_SCALE * (expf(x) - 1);
    }
    CLEAR_NET_ASSERT(0 && "Invalid Activation");
    return 0;
}

float dactf(float y, Activation act) {
    switch (act) {
    case Sigmoid:
        return y * (1 - y);
    case ReLU:
        return y > 0 ? 1.f : 0.f;
    case Leaky_ReLU:
        return y >= 0 ? 1 : CLEAR_NET_ACT_NEG_SCALE;
    case Tanh:
        return 1 - y * y;
    case ELU:
        // if result is negative then
        // y = CLEAR_NET_ACT_NEG_SCALE * (expf(x) - 1)
        // y = CLEAR_NET_ACT_NEG_SCALE(expf(x)) - CLEAR_NET_ACT_NEG_SCALE
        // y + CLEAR_NET_ACT_NEG_SLACE = CLEAR_NET_ACT_NEG_SCALE(expf(x))
        // otherwise it is 1
        return y > 0 ? 1 : y + CLEAR_NET_ACT_NEG_SCALE;
    }
    // TODO Don't think this is necessary
    CLEAR_NET_ASSERT(0 && "Invalid Activation");
    return 0.0f;
}

/************/
/* Matrices */
/************/
Matrix alloc_mat(size_t nrows, size_t ncols) {
    Matrix mat;
    mat.nrows = nrows;
    mat.ncols = ncols;
    mat.stride = ncols;
    mat.elements = CLEAR_NET_ALLOC(nrows * ncols * sizeof(*mat.elements));
    CLEAR_NET_ASSERT(mat.elements != NULL);
    return mat;
}

void dealloc_mat(Matrix *mat) {
    CLEAR_NET_DEALLOC(mat->elements);
    mat->elements = NULL;
    mat->nrows = 0;
    mat->ncols = 0;
    mat->stride = 0;
}

Matrix mat_form(size_t nrows, size_t ncols, size_t stride, float *elements) {
    return (Matrix){
        .nrows = nrows, .ncols = ncols, .stride = stride, .elements = elements};
}

void mat_print(Matrix mat, char *name) {
    printf("%s = [\n", name);
    for (size_t i = 0; i < mat.nrows; ++i) {
        printf("    ");
        for (size_t j = 0; j < mat.ncols; ++j) {
            printf("%f ", MAT_GET(mat, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}

Matrix mat_row(Matrix giver, size_t row) {
    return mat_form(1, giver.ncols, giver.stride, &MAT_GET(giver, row, 0));
}

void mat_copy(Matrix dest, Matrix giver) {
    CLEAR_NET_ASSERT(dest.nrows == giver.nrows);
    CLEAR_NET_ASSERT(dest.ncols == giver.ncols);
    for (size_t i = 0; i < giver.nrows; ++i) {
        for (size_t j = 0; j < giver.ncols; ++j) {
            MAT_GET(dest, i, j) = MAT_GET(giver, i, j);
        }
    }
}

float randf(void) { return (float)rand() / (float)RAND_MAX; }

void mat_rand(Matrix mat, float lower, float upper) {
    for (size_t i = 0; i < mat.nrows; ++i) {
        for (size_t j = 0; j < mat.ncols; ++j) {
            MAT_GET(mat, i, j) = randf() * (upper - lower) + lower;
        }
    }
}

void mat_sum(Matrix dest, Matrix toAdd) {
    CLEAR_NET_ASSERT(dest.nrows == toAdd.nrows);
    CLEAR_NET_ASSERT(dest.ncols == toAdd.ncols);
    for (size_t i = 0; i < dest.nrows; ++i) {
        for (size_t j = 0; j < dest.ncols; ++j) {
            MAT_GET(dest, i, j) += MAT_GET(toAdd, i, j);
        }
    }
}

void mat_act(Matrix mat, Activation act) {
    for (size_t i = 0; i < mat.nrows; ++i) {
        for (size_t j = 0; j < mat.ncols; ++j) {
            MAT_GET(mat, i, j) = actf(MAT_GET(mat, i, j), act);
        }
    }
}

void mat_fill(Matrix m, float x) {
    for (size_t i = 0; i < m.nrows; ++i) {
        for (size_t j = 0; j < m.ncols; ++j) {
            MAT_GET(m, i, j) = x;
        }
    }
}

void mat_mul(Matrix dest, Matrix left, Matrix right) {
    CLEAR_NET_ASSERT(left.ncols == right.nrows);
    CLEAR_NET_ASSERT(dest.nrows == left.nrows);
    CLEAR_NET_ASSERT(dest.ncols == right.ncols);

    mat_fill(dest, 0);

    for (size_t i = 0; i < dest.nrows; ++i) {
        for (size_t k = 0; k < left.ncols; ++k) {
            for (size_t j = 0; j < dest.ncols; ++j) {
                MAT_GET(dest, i, j) +=
                    MAT_GET(left, i, k) * MAT_GET(right, k, j);
            }
        }
    }
}

void mat_randomize_rows(Matrix mat) {
    for (size_t i = 0; i < mat.nrows; ++i) {
        size_t j = i + rand() % (mat.nrows - i);
        if (i != j) {
            for (size_t k = 0; k < mat.ncols; ++k) {
                float t = MAT_GET(mat, i, k);
                MAT_GET(mat, i, k) = MAT_GET(mat, j, k);
                MAT_GET(mat, j, k) = t;
            }
        }
    }
}

void mat_write_to_file(FILE *fp, Matrix mat) {
    for (size_t i = 0; i < mat.nrows; ++i) {
        for (size_t j = 0; j < mat.ncols; ++j) {
            fwrite(&MAT_GET(mat, i, j), sizeof(*mat.elements), 1, fp);
        }
    }
}

/*******/
/* Net */
/*******/
Net alloc_net(size_t *shape, size_t nlayers) {
    Net net;
    net.nlayers = nlayers;
    net.shape = shape;

    net.weights = CLEAR_NET_ALLOC(sizeof(*net.weights) * (net.nlayers - 1));
    CLEAR_NET_ASSERT(net.weights != NULL);

    net.weight_alters =
        CLEAR_NET_ALLOC(sizeof(*net.weight_alters) * (net.nlayers - 1));
    CLEAR_NET_ASSERT(net.weight_alters != NULL);

    net.biases = CLEAR_NET_ALLOC(sizeof(*net.biases) * (net.nlayers - 1));
    CLEAR_NET_ASSERT(net.biases != NULL);

    net.activations = CLEAR_NET_ALLOC(sizeof(*net.activations) * (net.nlayers));
    CLEAR_NET_ASSERT(net.activations != NULL);

    net.act_alters = CLEAR_NET_ALLOC(sizeof(*net.act_alters) * (net.nlayers));
    CLEAR_NET_ASSERT(net.act_alters != NULL);

    if (CLEAR_NET_MOMENTUM) {
        net.momentum_weight_store = CLEAR_NET_ALLOC(
            sizeof(*net.momentum_weight_store) * (net.nlayers - 1));
        CLEAR_NET_ASSERT(net.momentum_weight_store != NULL);

        net.momentum_bias_store = CLEAR_NET_ALLOC(
            sizeof(*net.momentum_bias_store) * (net.nlayers - 1));
        CLEAR_NET_ASSERT(net.momentum_bias_store != NULL);
    }

    // allocate the thing that will be the input
    // one row by the dimensions of the input
    net.activations[0] = alloc_mat(1, shape[0]);
    net.act_alters[0] = alloc_mat(1, shape[0]);
    // matrices are filled if they are read before set
    for (size_t i = 1; i < net.nlayers; ++i) {
        // allocate weights by the columns of previous activation and the
        // number of neurons of the this layer
        net.weights[i - 1] = alloc_mat(net.activations[i - 1].ncols, shape[i]);
        net.weight_alters[i - 1] =
            alloc_mat(net.activations[i - 1].ncols, shape[i]);
        mat_fill(net.weight_alters[i - 1], 0);
        if (CLEAR_NET_MOMENTUM) {
            net.momentum_weight_store[i - 1] =
                alloc_mat(net.activations[i - 1].ncols, shape[i]);
            mat_fill(net.momentum_weight_store[i - 1], 0);
            net.momentum_bias_store[i - 1] = alloc_mat(1, shape[i]);
            mat_fill(net.momentum_bias_store[i - 1], 0);
        }

        // allocate biases as one row and the shape of this layer
        net.biases[i - 1] = alloc_mat(1, shape[i]);
        // allocate activations as one row to add to each
        net.activations[i] = alloc_mat(1, shape[i]);
        net.act_alters[i] = alloc_mat(1, shape[i]);
    }
    return net;
}

void dealloc_net(Net *net, size_t shape_allocated) {
    // Deallocate matrices within each layer
    for (size_t i = 0; i < net->nlayers - 1; ++i) {
        dealloc_mat(&net->weights[i]);
        dealloc_mat(&net->weight_alters[i]);
        dealloc_mat(&net->biases[i]);
        dealloc_mat(&net->activations[i]);
        dealloc_mat(&net->act_alters[i]);
        // NOTE: Since shape is most likely created with {}
        // by users and is created with allocation when
        // reading from file, cannot consistently call
        // some deallocation on it
        if (CLEAR_NET_MOMENTUM) {
            dealloc_mat(&net->momentum_weight_store[i]);
            dealloc_mat(&net->momentum_bias_store[i]);
        }
    }

    // Deallocate the activation matrix of the output layer
    dealloc_mat(&net->activations[net->nlayers - 1]);
    dealloc_mat(&net->act_alters[net->nlayers - 1]);

    // Set net properties to NULL and 0
    net->nlayers = 0;
    net->activations = NULL;
    net->weights = NULL;
    net->weight_alters = NULL;
    net->momentum_weight_store = NULL;
    net->biases = NULL;
    if (shape_allocated) {
        CLEAR_NET_DEALLOC(net->shape);
        net->shape = NULL;
    }
}

void net_print(Net net, char *name) {
    char buf[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < net.nlayers - 1; ++i) {
        snprintf(buf, sizeof(buf), "weight matrix: %zu", i);
        mat_print(net.weights[i], buf);
        snprintf(buf, sizeof(buf), "bias matrix: %zu", i);
        mat_print(net.biases[i], buf);
    }
    printf("]\n");
}

void net_rand(Net net, float low, float high) {
    for (size_t i = 0; i < net.nlayers - 1; ++i) {
        mat_rand(net.weights[i], low, high);
        mat_rand(net.biases[i], low, high);
    }
}

void net_forward(Net net) {
    for (size_t i = 0; i < net.nlayers - 1; ++i) {
        // net.activations[0] stores the input, net.weghts[0] stores the first
        // weights
        mat_mul(net.activations[i + 1], net.activations[i], net.weights[i]);
        mat_sum(net.activations[i + 1], net.biases[i]);
        if (i == net.nlayers - 2) {
            mat_act(net.activations[i + 1], CLEAR_NET_ACT_OUTPUT);
        } else {
            mat_act(net.activations[i + 1], CLEAR_NET_ACT_HIDDEN);
        }
    }
}

/* Error */
float net_errorf(Net net, Matrix input, Matrix target) {
    CLEAR_NET_ASSERT(input.nrows == target.nrows);
    CLEAR_NET_ASSERT(target.ncols == NET_OUTPUT(net).ncols);

    float err = 0;
    size_t num_input = input.nrows;
    size_t dim_output = target.ncols;
    for (size_t i = 0; i < num_input; ++i) {
        Matrix x = mat_row(input, i);

        mat_copy(NET_INPUT(net), x);
        net_forward(net);

        for (size_t j = 0; j < dim_output; ++j) {
            float difference =
                MAT_GET(NET_OUTPUT(net), 0, j) - MAT_GET(target, i, j);
            err += difference * difference;
        }
    }
    return err / num_input;
}

void net_backprop(Net net, Matrix input, Matrix target) {
    CLEAR_NET_ASSERT(input.nrows == target.nrows);
    size_t num_i = input.nrows;
    CLEAR_NET_ASSERT(target.ncols == NET_OUTPUT(net).ncols);
    size_t dim_o = target.ncols;

    float coef = CLEAR_NET_RATE / num_i;
    // for each input
    for (size_t i = 0; i < num_i; ++i) {
        mat_copy(NET_INPUT(net), mat_row(input, i));
        net_forward(net);

        // for the output activation layer
        for (size_t j = 0; j < dim_o; ++j) {
            MAT_GET(net.act_alters[net.nlayers - 1], 0, j) =
                2 * (MAT_GET(NET_OUTPUT(net), 0, j) - MAT_GET(target, i, j));
        }

        // first layer is the output, make the changes to the one before itn
        for (size_t layer = net.nlayers - 1; layer > 0; --layer) {
            // this layers activation columns is the columns from its previous
            // matrix
            for (size_t j = 0; j < net.activations[layer].ncols; ++j) {
                float act = MAT_GET(net.activations[layer], 0, j);
                float dact;
                if (layer == net.nlayers - 1) {
                    dact = dactf(act, CLEAR_NET_ACT_OUTPUT);
                } else {
                    dact = dactf(act, CLEAR_NET_ACT_HIDDEN);
                }
                float alter = MAT_GET(net.act_alters[layer], 0, j);

                // biases are never read in backpropagation so their
                // change can be done in place
                size_t prev_layer = layer - 1;
                float change = alter * dact;
                if (CLEAR_NET_MOMENTUM) {
                    MAT_GET(net.momentum_bias_store[prev_layer], 0, j) =
                        CLEAR_NET_MOMENTUM_BETA *
                            MAT_GET(net.momentum_bias_store[prev_layer], 0, j) +
                        (1 - CLEAR_NET_MOMENTUM_BETA) * change;
                    change = MAT_GET(net.momentum_bias_store[prev_layer], 0, j);
                }
                MAT_GET(net.biases[prev_layer], 0, j) -= coef * change;

                // this activations columns is equal to the rows of its next
                // matrix
                for (size_t k = 0; k < net.activations[prev_layer].ncols; ++k) {
                    float prev_act = MAT_GET(net.activations[prev_layer], 0, k);
                    float prev_weight = MAT_GET(net.weights[prev_layer], k, j);
                    MAT_GET(net.act_alters[prev_layer], 0, k) +=
                        alter * dact * prev_weight;
                    MAT_GET(net.weight_alters[prev_layer], k, j) +=
                        alter * dact * prev_act;
                }
            }
        }

        // reset for next iteration
        for (size_t j = 0; j < net.nlayers; ++j) {
            mat_fill(net.act_alters[j], 0);
        }
    }

    for (size_t i = 0; i < net.nlayers - 1; ++i) {
        for (size_t j = 0; j < net.weights[i].nrows; ++j) {
            for (size_t k = 0; k < net.weights[i].ncols; ++k) {
                float change = MAT_GET(net.weight_alters[i], j, k);
                if (CLEAR_NET_MOMENTUM) {
                    MAT_GET(net.momentum_weight_store[i], j, k) =
                        CLEAR_NET_MOMENTUM_BETA *
                            MAT_GET(net.momentum_weight_store[i], j, k) +
                        (1 - CLEAR_NET_MOMENTUM_BETA) * change;
                    change = MAT_GET(net.momentum_weight_store[i], j, k);
                }
                MAT_GET(net.weights[i], j, k) -= coef * change;

                // reset for next backpropagation
                MAT_GET(net.weight_alters[i], j, k) = 0;
            }
        }
    }
}

void net_get_batch(Matrix *batch_input, Matrix *batch_output, Matrix input,
                   Matrix output, size_t batch_num, size_t batch_size) {
    *batch_input = mat_form(batch_size, input.ncols, input.stride,
                            &MAT_GET(input, batch_num * batch_size, 0));
    *batch_output = mat_form(batch_size, output.ncols, output.stride,
                             &MAT_GET(output, batch_num * batch_size, 0));
}

void net_print_results(Net net, Matrix input, Matrix target,
                       float (*fix_output)(float)) {
    CLEAR_NET_ASSERT(input.nrows == target.nrows);
    CLEAR_NET_ASSERT(NET_OUTPUT(net).ncols == target.ncols);
    size_t num_i = input.nrows;
    size_t dim_i = input.ncols;
    size_t dim_o = target.ncols;

    printf("Final Cost: %f\n", net_errorf(net, input, target));
    printf("Input | Target | Prediction\n");
    for (size_t i = 0; i < num_i; ++i) {
        Matrix in = mat_row(input, i);
        mat_copy(NET_INPUT(net), in);
        net_forward(net);
        for (size_t j = 0; j < dim_i; ++j) {
            printf("%f ", MAT_GET(input, i, j));
        }
        printf(" | ");
        for (size_t j = 0; j < dim_o; ++j) {
            printf("%f ", fix_output(MAT_GET(target, i, j)));
        }
        printf(" | ");
        for (size_t j = 0; j < dim_o; ++j) {
            printf("%f ", fix_output(MAT_GET(NET_OUTPUT(net), 0, j)));
        }
        printf("\n");
    }
}

void net_save_to_file(char *file_name, Net net) {
    // with a shape and a list of floats the entire model can be derived
    FILE *fp;
    fp = fopen(file_name, "wb");
    fwrite(&net.nlayers, sizeof(net.nlayers), 1, fp);
    for (size_t i = 0; i < net.nlayers; ++i) {
        fwrite(&net.shape[i], sizeof(*net.shape), 1, fp);
    }

    for (size_t i = 0; i < net.nlayers - 1; ++i) {
        mat_write_to_file(fp, net.weights[i]);
        mat_write_to_file(fp, net.biases[i]);
    }
    fclose(fp);
}

Net alloc_net_from_file(char *file_name) {
    FILE *fp = fopen(file_name, "rb");
    if (fp == NULL) {
        fclose(fp);
    }
    CLEAR_NET_ASSERT(fp != NULL);

    size_t nlayers = 0;
    fread(&nlayers, sizeof(nlayers), 1, fp);

    // if it is less than two than there is no io
    CLEAR_NET_ASSERT(nlayers >= 2);

    size_t *shape = (size_t *)CLEAR_NET_ALLOC(nlayers * sizeof(nlayers));
    CLEAR_NET_ASSERT(shape != NULL);
    fread(shape, sizeof(*shape), nlayers, fp);
    Net net = alloc_net(shape, nlayers);
    size_t num_rows;
    size_t num_cols;
    Matrix weight;
    Matrix bias;
    for (size_t layer = 0; layer < net.nlayers - 1; ++layer) {
        weight = net.weights[layer];
        num_rows = weight.nrows;
        num_cols = weight.ncols;
        for (size_t j = 0; j < num_rows; ++j) {
            for (size_t k = 0; k < num_cols; ++k) {
                // check if i can read numcols
                fread(&MAT_GET(weight, j, k), sizeof(*weight.elements), 1, fp);
            }
        }
        bias = net.biases[layer];
        for (size_t k = 0; k < num_cols; ++k) {
            // check if i can read numcols
            fread(&MAT_GET(bias, 0, k), sizeof(*bias.elements), 1, fp);
        }
    }

    fclose(fp);
    return net;
}

#endif // CLEAR_NET_IMPLEMENTATION
