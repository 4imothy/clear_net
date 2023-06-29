// TODO implement RELU for hidden functions to combat vanishing gradient
// TODO function to print inputs vs target vs predicted
// TODO implement save and loading the model
// TODO example on iris, other math functions
// TODO benchmark againts other neural nets with time and memory used

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

#ifndef CLEAR_NET_RATE
#define CLEAR_NET_RATE 1.0f
#endif
#ifndef CLEAR_NET_ACT_OUTPUT
#define CLEAR_NET_ACT_OUTPUT Sigmoid
#endif // CLEAR_NET_ACT_OUTPUT
#ifndef CLEAR_NET_ACT_HIDDEN
#define CLEAR_NET_ACT_HIDDEN LEAKY_RELU
#endif // CLEAR_NET_ACT_HIDDEN
#ifndef CLEAR_NET_LEAKY_RELU_NEG_SCALE
#define CLEAR_NET_LEAKY_RELU_NEG_SCALE 0.01f
#endif // CLEAR_NET_LEAKY_RELU_NEG_SCALE

/*
Below are the definitions of structs and enums and the
declaractions of functions that are defined later.
Some functions are commented out to abstract and
keep users' namespace sane.
*/

// float randf();

/* Activation functions */
// Can add tanh, elu
// float reluf(float x);
// float actf(float x, Activation act);
// float dactf(float y, Activation act);

typedef enum {
    Sigmoid,
	RELU,
	LEAKY_RELU,
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
// void mat_mul(Matrix dest, Matrix left, Matrix right);
// void mat_sum(Matrix dest, Matrix toAdd);
// void mat_rand(Matrix mat, float lower, float upper);
// void mat_act(Matrix mat);

/* Net */
typedef struct {
    size_t nlayers;
    Matrix *activations;
    // number of these is equal to the number of layers -1 (for the output)
    Matrix *weights;
    Matrix *biases;
    // this stores the changes to be done to the weihts
    Matrix *weight_alters;
    // stores changes in activation results
    Matrix *activation_alters;
    size_t *shape;
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
void net_backprop(Net net, Matrix input, Matrix target);
void net_print_results(Net net, Matrix input, Matrix target);
// void net_forward(Net net);

#endif // CLEAR_NET

#ifdef CLEAR_NET_IMPLEMENTATION

/* Activation Functions */
float actf(float x, Activation act) {
  switch (act) {
    case Sigmoid:
	    return 1.f / (1.f + expf(-x));
    case RELU:
    	return x > 0 ? x : 0.f;
  case LEAKY_RELU:
	return x >= 0 ? x : CLEAR_NET_LEAKY_RELU_NEG_SCALE * x;
    }
    CLEAR_NET_ASSERT(0 && "Invalid Activation");
    return 0.0f;
}

float dactf(float y, Activation act) {
    switch (act) {
    case Sigmoid:
        return y * (1 - y);
	case RELU:
	  return y > 0 ? 1 : 0.f;
	case LEAKY_RELU:
	  return y >= 0 ? 1 : CLEAR_NET_LEAKY_RELU_NEG_SCALE;
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

    net.activation_alters =
        CLEAR_NET_ALLOC(sizeof(*net.activation_alters) * (net.nlayers));
    CLEAR_NET_ASSERT(net.activation_alters != NULL);

    // allocate the thing that will be the input
    // one row by the dimensions of the input
    net.activations[0] = alloc_mat(1, shape[0]);
    net.activation_alters[0] = alloc_mat(1, shape[0]);
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
        net.activation_alters[i] = alloc_mat(1, shape[i]);
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

        // number outputs is the ncols of target
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
            MAT_GET(net.activation_alters[net.nlayers - 1], 0, j) =
                2 * (MAT_GET(NET_OUTPUT(net), 0, j) - MAT_GET(target, i, j));
        }

        // first layer is the output, make the changes to the one before it
        for (size_t l = net.nlayers - 1; l > 0; --l) {
            // this layers activation columns is the columns from its previous
            // matrix
            for (size_t j = 0; j < net.activations[l].ncols; ++j) {
                float a = MAT_GET(net.activations[l], 0, j);
                float da = MAT_GET(net.activation_alters[l], 0, j);
				float qa;
    	     	qa = dactf(a, CLEAR_NET_ACT_HIDDEN);
                // biases are never read in backpropagation so their
                // change can be done in place
                MAT_GET(net.biases[l - 1], 0, j) -= coef * da * qa;

                // this activations columns is equal to the rows of its next
                // matrix
                for (size_t k = 0; k < net.activations[l - 1].ncols; ++k) {
                    float pa = MAT_GET(net.activations[l - 1], 0, k);
                    float w = MAT_GET(net.weights[l - 1], k, j);
                    MAT_GET(net.activation_alters[l - 1], 0, k) += da * qa * w;
                    MAT_GET(net.weight_alters[l - 1], k, j) +=
                        coef * da * qa * pa;
                }
            }
        }
        // reset for next iteration
        for (size_t j = 0; j < net.nlayers; ++j) {
            mat_fill(net.activation_alters[j], 0);
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

void net_print_results(Net net, Matrix input, Matrix target) {
  CLEAR_NET_ASSERT(input.nrows == target.nrows);
  CLEAR_NET_ASSERT(NET_OUTPUT(net).ncols == target.ncols);
  size_t num_i = input.nrows;
  size_t dim_i = input.ncols;
  size_t dim_o = target.ncols;

  printf("Final Cost: %f\n", net_errorf(net, input, target));
  printf("Input | Prediction | Target\n");
  for (size_t i = 0; i < num_i; ++i) {
    Matrix in = mat_row(input, i);
    mat_copy(NET_INPUT(net), in);
    net_forward(net);
    for (size_t j = 0; j < dim_i; ++j) {
      printf("%f ", MAT_GET(input, i, j));
    }
    printf(" | ");
    for (size_t j = 0; j < dim_o; ++j) {
      printf("%f ", MAT_GET(target, i, j));
    }
    printf(" | ");
    for (size_t j = 0; j < dim_o; ++j) {
      printf("%f ", MAT_GET(NET_OUTPUT(net), 0, j));
    }
    printf("\n");
  }
}

#endif // CLEAR_NET_IMPLEMENTATION
