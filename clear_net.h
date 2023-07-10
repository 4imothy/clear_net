// TODO implement strassen and other faster matrix multiplication algorithms
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

/*
Below are the definitions of structs and enums and the
declaractions of functions that are defined later.
Some functions are commented out to abstract and
keep users' namespace sane.
*/

// float randf(void);

/* Activation functions */
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

/* Matrices */
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
// void mat_write_to_file(FILE *f, Matrix mat);
// void mat_mul(Matrix dest, Matrix left, Matrix right);
// void mat_sum(Matrix dest, Matrix toAdd);
// void mat_rand(Matrix mat, float lower, float upper);
// void mat_act(Matrix mat);

/* Net
Basic Structure
Each layer consists of weights -> biases -> activation.
Each forward takes the previous activation matrix multiply
with the weights, add the bias apply correct activation function.
Each column in the weight matrix is a neuron.
Each activation * weights results in a single column matrix
which the bias is added to.
*/
typedef struct {
    size_t nlayers;
    Matrix *activations;
    // number of these is equal to the number of layers -1 (for the output)
    Matrix *weights;
    Matrix *biases;
    // this stores the changes to be done to the weihts
    Matrix *weight_alters;
    // stores changes in activation results
    Matrix *buffer;
    size_t *shape;
} Net;

#define NET_INPUT(net)                                                         \
    (CLEAR_NET_ASSERT((net).nlayers > 0), (net).activations[0])
#define NET_OUTPUT(net)                                                        \
    (CLEAR_NET_ASSERT((net).nlayers > 0), (net).activations[(net).nlayers - 1])
#define NET_PRINT(net) net_print(net, #net)

Net alloc_net(size_t *shape, size_t nlayers);
void dealloc_net(Net *net);
Net alloc_net_from_file(char *file_name);
void net_save_to_file(char *file_name, Net net);
float net_errorf(Net net, Matrix input, Matrix target);
void net_print(Net net, char *name);
void net_rand(Net net, float low, float high);
void net_backprop(Net net, Matrix input, Matrix target);
void net_print_results(Net net, Matrix input, Matrix target,
                       float (*fix_output)(float));
// void net_forward(Net net);

#endif // CLEAR_NET

#ifdef CLEAR_NET_IMPLEMENTATION

/* Activation Functions */
float actf(float x, Activation act) {
    switch (act) {
    case Sigmoid:
        return 1.f / (1.f + expf(-x));
    case ReLU:
        return x > 0 ? x : 0.f;
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
    return 0.0f;
}

float dactf(float y, Activation act) {
    switch (act) {
    case Sigmoid:
        return y * (1 - y);
    case ReLU:
        return y > 0 ? 1 : 0.f;
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
    CLEAR_NET_ASSERT(0 && "Invalid Activation");
    return 0.0f;
}

/* Matrices */
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

void mat_write_to_file(FILE *fp, Matrix mat) {
    for (size_t i = 0; i < mat.nrows; ++i) {
        for (size_t j = 0; j < mat.ncols; ++j) {
            fwrite(&MAT_GET(mat, i, j), sizeof(*mat.elements), 1, fp);
        }
    }
}

/* Net */
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

    net.buffer = CLEAR_NET_ALLOC(sizeof(*net.buffer) * (net.nlayers));
    CLEAR_NET_ASSERT(net.buffer != NULL);

    // allocate the thing that will be the input
    // one row by the dimensions of the input
    net.activations[0] = alloc_mat(1, shape[0]);
    net.buffer[0] = alloc_mat(1, shape[0]);
    for (size_t i = 1; i < net.nlayers; ++i) {
        // allocate weights by the columns of previous activation and the
        // number of neurons of the this layer
        net.weights[i - 1] = alloc_mat(net.activations[i - 1].ncols, shape[i]);
        net.weight_alters[i - 1] =
            alloc_mat(net.activations[i - 1].ncols, shape[i]);
        // allocate biases as one row and the shape of this layer
        net.biases[i - 1] = alloc_mat(1, shape[i]);
        // allocate activations as one row to add to each
        net.activations[i] = alloc_mat(1, shape[i]);
        net.buffer[i] = alloc_mat(1, shape[i]);
    }
    return net;
}

void dealloc_net(Net *net) {
    // Deallocate matrices within each layer
    for (size_t i = 0; i < net->nlayers - 1; ++i) {
        dealloc_mat(&net->weights[i]);
        dealloc_mat(&net->biases[i]);
        dealloc_mat(&net->activations[i]);
        dealloc_mat(&net->buffer[i]);
        dealloc_mat(&net->weight_alters[i]);
        // NOTE: Since shape is most likely created with {}
        // by users and is created with allocation when
        // reading from file, cannot consistently call
        // some deallocation on it
        net->shape[i] = 0;
    }

    // Deallocate the activation matrix of the output layer
    dealloc_mat(&net->activations[net->nlayers - 1]);
    dealloc_mat(&net->buffer[net->nlayers - 1]);

    // Set net properties to NULL and 0
    net->nlayers = 0;
    net->activations = NULL;
    net->weights = NULL;
    net->biases = NULL;
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
            MAT_GET(net.buffer[net.nlayers - 1], 0, j) =
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
                float alter_act = MAT_GET(net.buffer[layer], 0, j);

                // biases are never read in backpropagation so their
                // change can be done in place
                size_t prev_layer = layer - 1;
                MAT_GET(net.biases[prev_layer], 0, j) -=
                    coef * alter_act * dact;

                // this activations columns is equal to the rows of its next
                // matrix
                for (size_t k = 0; k < net.activations[prev_layer].ncols; ++k) {
                    float prev_act = MAT_GET(net.activations[prev_layer], 0, k);
                    float prev_weight = MAT_GET(net.weights[prev_layer], k, j);
                    MAT_GET(net.buffer[prev_layer], 0, k) +=
                        alter_act * dact * prev_weight;
                    MAT_GET(net.weight_alters[prev_layer], k, j) +=
                        coef * alter_act * dact * prev_act;
                }
            }
        }

        // reset for next iteration
        for (size_t j = 0; j < net.nlayers; ++j) {
            mat_fill(net.buffer[j], 0);
        }
    }

    for (size_t i = 0; i < net.nlayers - 1; ++i) {
        for (size_t j = 0; j < net.weights[i].nrows; ++j) {
            for (size_t k = 0; k < net.weights[i].ncols; ++k) {
                MAT_GET(net.weights[i], j, k) -=
                    MAT_GET(net.weight_alters[i], j, k);
                // reset for next backpropagation
                MAT_GET(net.weight_alters[i], j, k) = 0;
            }
        }
    }
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
