// implement RELU for hidden functions to combat vanishing gradient
#ifndef CLEAR_NET
#define CLEAR_NET

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

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
#include "assert.h"
#define CLEAR_NET_ASSERT assert
#endif // CLEAR_NET_ASSERT

#ifndef CLEAR_NET_ACT
#define CLEAR_NET_ACT Sigmoid
#endif // CLEAR_NET_ACT

/*
Below are the definitions of structs and enums and the
declaractions of functions that are defined later.
Some functions are commented out to abstract and
keep users' namespace sane.
*/

// float randf();
// float actf(float x);
// float dactf(float y);

/* Activation functions */
// float sigmoidf(float x);

typedef enum {
    Sigmoid,
} Activations;

// Matrices
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
// void mat_mul(Matrix dest, Matrix left, Matrix right);
// void mat_sum(Matrix dest, Matrix toAdd);
// void mat_rand(Matrix mat, float lower, float upper);
// void mat_act(Matrix mat);

// Net
typedef struct {
    size_t nlayers;
    Matrix *activations;
    // number of these is equal to the number of layers -1 (for the output)
    Matrix *weights;
    Matrix *biases;
} Net;

#define NET_INPUT(net)                                                         \
    (CLEAR_NET_ASSERT((net).nlayers > 0), (net).activations[0])
#define NET_OUTPUT(net)                                                        \
    (CLEAR_NET_ASSERT((net).nlayers > 0), (net).activations[(net).nlayers - 1])
#define NET_PRINT(net) net_print(net, #net)

Net alloc_net(size_t *shape, size_t nlayers);
void dealloc_net(Net *net);
float net_errorf(Net net, Matrix input, Matrix target);
void net_print(Net net, char *name);
void net_rand(Net net, float low, float high);
void net_backprop(Net net, Matrix input, Matrix output);
// void net_forward(Net net);

/* Error functions */
// float mean_squaredf(Net net, Matrix input, Matrix output);

#endif // CLEAR_NET

#ifdef CLEAR_NET_IMPLEMENTATION

// Activation functions
float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

// Matrices

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

float randf() { return (float)rand() / (float)RAND_MAX; }

void mat_rand(Matrix mat, float lower, float upper) {
    for (size_t i = 0; i < mat.nrows; ++i) {
        for (size_t j = 0; j < mat.ncols; ++j) {
            MAT_GET(mat, i, j) = randf() * (upper - lower) + lower;
        }
    }
}

void mat_mul(Matrix dest, Matrix left, Matrix right) {
    // ensure that the inner dimensions are equal
    CLEAR_NET_ASSERT(left.ncols == right.nrows);
    size_t inner = left.ncols;
    // assert that destination has the outer dimensions
    CLEAR_NET_ASSERT(dest.nrows == left.nrows);
    CLEAR_NET_ASSERT(dest.ncols == right.ncols);
    // iterate over outer size
    for (size_t i = 0; i < left.nrows; ++i) {
        for (size_t j = 0; j < right.ncols; ++j) {
            MAT_GET(dest, i, j) = 0;
            // iterater over the inner size
            for (size_t k = 0; k < inner; ++k) {
                MAT_GET(dest, i, j) +=
                    MAT_GET(left, i, k) * MAT_GET(right, k, j);
            }
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

float actf(float x) {
    switch (CLEAR_NET_ACT) {
    case Sigmoid:
        return sigmoidf(x);
    }
    CLEAR_NET_ASSERT(0 && "Invalid Activation");
    return 0.0f;
}

float dactf(float y) {
    switch (CLEAR_NET_ACT) {
    case Sigmoid:
        return y * (1 - y);
    }
    CLEAR_NET_ASSERT(0 && "Invalid Activation");
    return 0.0f;
}

void mat_act(Matrix mat) {
    for (size_t i = 0; i < mat.nrows; ++i) {
        for (size_t j = 0; j < mat.ncols; ++j) {
            MAT_GET(mat, i, j) = actf(MAT_GET(mat, i, j));
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

// Net functions
Net alloc_net(size_t *shape, size_t nlayers) {
    Net net;
    net.nlayers = nlayers;

    net.weights = CLEAR_NET_ALLOC(sizeof(*net.weights) * (net.nlayers - 1));
    CLEAR_NET_ASSERT(net.weights != NULL);

    net.biases = CLEAR_NET_ALLOC(sizeof(*net.biases) * (net.nlayers - 1));
    CLEAR_NET_ASSERT(net.biases != NULL);

    net.activations = CLEAR_NET_ALLOC(sizeof(*net.activations) * (net.nlayers));
    CLEAR_NET_ASSERT(net.activations != NULL);

    // allocate the thing that will be the input
    // one row by the dimensions of the input
    net.activations[0] = alloc_mat(1, shape[0]);
    for (size_t i = 1; i < net.nlayers; ++i) {
        // allocate weights by the columns of previous activation and the
        // number of neurons of the this layer
        net.weights[i - 1] = alloc_mat(net.activations[i - 1].ncols, shape[i]);
        // allocate biases as one row and the shape of this layer
        net.biases[i - 1] = alloc_mat(1, shape[i]);
        // allocate activations as one row to add to each
        net.activations[i] = alloc_mat(1, shape[i]);
    }
    return net;
}

void dealloc_net(Net *net) {
    // Deallocate matrices within each layer
    for (size_t i = 0; i < net->nlayers; ++i) {
        dealloc_mat(&net->weights[i]);
        dealloc_mat(&net->biases[i]);
        dealloc_mat(&net->activations[i]);
    }

    // Deallocate the activation matrix of the output layer
    dealloc_mat(&net->activations[net->nlayers]);

    // Deallocate the arrays of matrices
    CLEAR_NET_DEALLOC(net->weights);
    CLEAR_NET_DEALLOC(net->biases);
    CLEAR_NET_DEALLOC(net->activations);

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
    // there is one more activation than there are layers
    for (size_t i = 0; i < net.nlayers - 1; ++i) {
        // the first activation is the input so we don't set that here
        mat_mul(net.activations[i + 1], net.activations[i], net.weights[i]);
        mat_sum(net.activations[i + 1], net.biases[i]);
        mat_act(net.activations[i + 1]);
    }
}

// Error functions
float mean_squaredf(Net net, Matrix input, Matrix target) {
    CLEAR_NET_ASSERT(input.nrows == target.nrows);
    CLEAR_NET_ASSERT(target.ncols == NET_OUTPUT(net).ncols);

    float err = 0;
    size_t num_input = input.nrows;
    size_t dim_output = target.ncols;
    for (size_t i = 0; i < num_input; ++i) {
        Matrix x = mat_row(input, i);

        mat_copy(NET_INPUT(net), x);
        net_forward(net);

        // number outputs is the ncols of target
        for (size_t j = 0; j < dim_output; ++j) {
            float difference =
                MAT_GET(NET_OUTPUT(net), 0, j) - MAT_GET(target, i, j);
            err += difference * difference;
        }
    }

    return err / num_input;
}

float net_errorf(Net net, Matrix input, Matrix target) {
    return mean_squaredf(net, input, target);
}

void net_backprop(Net net, Matrix input, Matrix target) {
    //	printf("here\n");
    CLEAR_NET_ASSERT(input.nrows == target.nrows);
    size_t num_i = input.nrows;
    CLEAR_NET_ASSERT(target.ncols == NET_OUTPUT(net).ncols);
    size_t dim_o = target.ncols;
    // size_t shape[] = {2, 2, 1};
    size_t shape[] = {3, 3, 8, 2};

    // output actually should be included in nlayers so do that
    // and doesn't need the +1, just allocate an
    // extra for the activations

    Net g = alloc_net(shape, net.nlayers);

    CLEAR_NET_ASSERT(net.nlayers == g.nlayers);

    // fill the biases, weights and activations
    for (size_t i = 0; i < g.nlayers - 1; ++i) {
        mat_fill(g.weights[i], 0);
        mat_fill(g.biases[i], 0);
        mat_fill(g.activations[i], 0);
    }
    mat_fill(g.activations[g.nlayers], 0);

    // one more for the output layer
    mat_fill(g.activations[g.nlayers - 1], 0);

    for (size_t i = 0; i < num_i; ++i) {
        mat_copy(NET_INPUT(net), mat_row(input, i));
        net_forward(net);

        for (size_t j = 0; j < net.nlayers; ++j) {
            mat_fill(g.activations[j], 0);
        }

        for (size_t j = 0; j < dim_o; ++j) {
            MAT_GET(NET_OUTPUT(g), 0, j) =
                2 * (MAT_GET(NET_OUTPUT(net), 0, j) - MAT_GET(target, i, j));
        }

        // first is just the output layer, then the hidden
        // nlayers includes the first to the one before the output layer
        for (size_t l = net.nlayers - 1; l > 0; --l) {
            // this layers activation columns is the columns from the previous
            // matrix
            for (size_t j = 0; j < net.activations[l].ncols; ++j) {
                float a = MAT_GET(net.activations[l], 0, j);
                float da = MAT_GET(g.activations[l], 0, j);
                float qa = dactf(a);
                MAT_GET(g.biases[l - 1], 0, j) += da * qa;

                // this activations columns is equal to the rows of the next
                // matrix
                for (size_t k = 0; k < net.activations[l - 1].ncols; ++k) {
                    float pa = MAT_GET(net.activations[l - 1], 0, k);
                    float w = MAT_GET(net.weights[l - 1], k, j);
                    MAT_GET(g.weights[l - 1], k, j) += da * qa * pa;
                    MAT_GET(g.activations[l - 1], 0, k) += da * qa * w;
                }
            }
        }
    }

    for (size_t i = 0; i < g.nlayers - 1; ++i) {
        for (size_t j = 0; j < g.weights[i].nrows; ++j) {
            for (size_t k = 0; k < g.weights[i].ncols; ++k) {
                MAT_GET(g.weights[i], j, k) /= num_i;
            }
        }
        for (size_t j = 0; j < g.biases[i].nrows; ++j) {
            for (size_t k = 0; k < g.biases[i].ncols; ++k) {
                MAT_GET(g.biases[i], j, k) /= num_i;
            }
        }
    }

    // apply the learning
    float rate = 1.0f;

    for (size_t i = 0; i < net.nlayers - 1; ++i) {
        for (size_t j = 0; j < net.weights[i].nrows; ++j) {
            for (size_t k = 0; k < net.weights[i].ncols; ++k) {
                MAT_GET(net.weights[i], j, k) -=
                    rate * MAT_GET(g.weights[i], j, k);
            }
        }

        for (size_t j = 0; j < net.biases[i].nrows; ++j) {
            for (size_t k = 0; k < net.biases[i].ncols; ++k) {
                MAT_GET(net.biases[i], j, k) -=
                    rate * MAT_GET(g.biases[i], j, k);
            }
        }
    }
    //	dealloc_net(&g);
}

#endif // CLEAR_NET_IMPLEMENTATION
