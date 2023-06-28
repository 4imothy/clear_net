#ifndef CLEAR_NET
#define CLEAR_NET

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define ARR_LEN(a) sizeof((a)) / sizeof((a))[0]
// allow custom memory allocation strategies
#ifndef CLEAR_NET_ALLOC
#define CLEAR_NET_ALLOC malloc
#endif // CLEAR_NET_MALLOC
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
#ifndef CLEAR_NET_ERR
#define CLEAR_NET_ERR MeanSquared
#endif // CLEAR_NET_ERR

// Activation functions
float sigmoidf(float x);
typedef enum {
    Sigmoid,
} Activations;

// Error functions
typedef enum { MeanSquared } Errors;

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
Matrix form_mat(size_t nrows, size_t ncols, size_t stride, float *elements);
void mat_print(Matrix mat, char *name);
void mat_mul(Matrix dest, Matrix left, Matrix right);
void mat_sum(Matrix dest, Matrix toAdd);
void mat_rand(Matrix mat, float lower, float upper);

// Net
typedef struct {
    size_t nlayers;
    Matrix *weights;
    Matrix *biases;
    Matrix *activations;
} Net;

#define NET_INPUT(net) (net).activations[0]
#define NET_OUTPUT(net) (net).activations[(net).nlayers]
#define NET_PRINT(net) net_print(net, #net)

Net alloc_net(size_t *shape, size_t nlayers);
void dealloc_net(Net *net);
float net_errorf(Net net, Matrix input, Matrix target);
void net_print(Net net, char *name);
void net_rand(Net net, float low, float high);
void net_backprop(Net net, Matrix input, Matrix output);

#endif // CLEAR_NET

#ifdef CLEAR_NET_IMPLEMENTATION

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

Matrix form_mat(size_t nrows, size_t ncols, size_t stride, float *elements) {
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
    return form_mat(1, giver.ncols, giver.stride, &MAT_GET(giver, row, 0));
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
        // the y is the result of a sigmoid already so don't need to call it
        // again
        return y * (1 - y);
    }
    CLEAR_NET_ASSERT(0 && "Invalid Activation");
    return 0.0f;
}

float mat_actf(Matrix mat) {
    for (size_t i = 0; i < mat.nrows; ++i) {
        for (size_t j = 0; j < mat.ncols; ++j) {
            MAT_GET(mat, i, j) = actf(MAT_GET(mat, i, j));
        }
    }
    return 0.0f;
}

// Activation functions
float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

// Net
Net alloc_net(size_t *shape, size_t nlayers) {
    Net net;
    net.nlayers = nlayers - 1;

    net.weights = CLEAR_NET_ALLOC(sizeof(*net.weights) * net.nlayers);
    CLEAR_NET_ASSERT(net.weights != NULL);

    net.biases = CLEAR_NET_ALLOC(sizeof(*net.biases) * net.nlayers);
    CLEAR_NET_ASSERT(net.biases != NULL);

    net.activations =
        CLEAR_NET_ALLOC(sizeof(*net.activations) * (net.nlayers + 1));
    CLEAR_NET_ASSERT(net.activations != NULL);

	// allocate the thing that will be the input
	// one row by the dimensions of the input
    net.activations[0] = alloc_mat(1, shape[0]);
    for (size_t i = 0; i < nlayers - 1; ++i){
	  // each weight is a matrix where each column is a neuron
        net.weights[i] = alloc_mat(net.activations[i].ncols, shape[i + 1]);
		// some dimensions as the num columns of the weight matrix
		// a scalar for each neuron
        net.biases[i] = alloc_mat(1, shape[i + 1]);
		// prepare for the next layer
        net.activations[i + 1] = alloc_mat(1, shape[i + 1]);
    }

    return net;
}

void dealloc_net(Net *net) {
    for (size_t i = 0; i < net->nlayers - 1; ++i) {
        dealloc_mat(&net->weights[i]);
        dealloc_mat(&net->biases[i]);
        dealloc_mat(&net->activations[i + 1]);
    }
    dealloc_mat(&net->activations[0]);

    CLEAR_NET_DEALLOC(net->weights);
    CLEAR_NET_DEALLOC(net->biases);
    CLEAR_NET_DEALLOC(net->activations);
}

void net_print(Net net, char *name) {
    char buf[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < net.nlayers; ++i) {
        snprintf(buf, sizeof(buf), "weight matrix: %zu", i);
        mat_print(net.weights[i], buf);
        snprintf(buf, sizeof(buf), "bias matrix: %zu", i);
        mat_print(net.biases[i], buf);
    }
    printf("]\n");
}

void net_rand(Net net, float low, float high) {
    for (size_t i = 0; i < net.nlayers; ++i) {
        mat_rand(net.weights[i], low, high);
        mat_rand(net.biases[i], low, high);
    }
}

void net_forward(Net net) {
    // there is one more activation than there are layers
    for (size_t i = 0; i < net.nlayers; ++i) {
        mat_mul(net.activations[i + 1], net.activations[i], net.weights[i]);
        mat_sum(net.activations[i + 1], net.biases[i]);
        mat_actf(net.activations[i + 1]);
    }
}

// Error functions
float mean_squaredf(Net net, Matrix input, Matrix target) {
    CLEAR_NET_ASSERT(input.nrows == target.nrows);
    CLEAR_NET_ASSERT(target.ncols == NET_OUTPUT(net).ncols);

    float err = 0;
    size_t num_outputs = input.nrows;
    for (size_t i = 0; i < input.nrows; ++i) {
        Matrix x = mat_row(input, i);

        mat_copy(NET_INPUT(net), x);
        net_forward(net);

        // number outputs is the ncols of target
        for (size_t j = 0; j < num_outputs; ++j) {
            float t = MAT_GET(NET_OUTPUT(net), 0, j) - MAT_GET(target, i, j);
            err += t * t;
        }
    }
    return err / num_outputs;
}

float net_errorf(Net net, Matrix input, Matrix target) {
    switch (CLEAR_NET_ERR) {
    case MeanSquared:
        return mean_squaredf(net, input, target);
    }
}

void net_backprop(Net net, Matrix input, Matrix target) {
    CLEAR_NET_ASSERT(input.nrows == target.nrows);
    CLEAR_NET_ASSERT(target.ncols == NET_OUTPUT(net).ncols);

    size_t num_io = input.nrows;
    size_t dim_output = target.ncols;

	// coefficient found in derivative
    float coef = 2.0f / num_io;

	// for each input
    for (size_t i = 0; i < num_io; ++i) {
        mat_copy(NET_INPUT(net), mat_row(input, i));
        net_forward(net);

		float prev_act_change;
		
		// for each dimension of the output
        for (size_t j = 0; j < dim_output; ++j) {
            float pred_out = MAT_GET(NET_OUTPUT(net), 0, j);
            float change = (pred_out - MAT_GET(target, i, j)) * dactf(pred_out);
            int change_layer = net.nlayers - 1;
			
            MAT_GET(net.biases[change_layer], 0, j) -= change;

			// for each neuron of the output layer
            for (size_t k = 0; k < net.weights[change_layer].ncols; ++k) {
		         float weight_change =
                    change * MAT_GET(net.activations[change_layer], 0, j);
				 float act_change = change * MAT_GET(net.weights[change_layer], j, k);
				
                MAT_GET(net.weights[change_layer], k, j) -= coef * weight_change;
				prev_act_change = act_change;
                MAT_GET(net.activations[change_layer], 0, j) -= coef * act_change;
            }
        }
        // for each of the hidden layers
        for (size_t layer_id = net.nlayers - 1; layer_id > 0; --layer_id) {

          for (size_t j = 0; j < net.weights[layer_id].ncols; ++j) {
            float act_out = MAT_GET(net.activations[layer_id], 0, j);
			float change = prev_act_change;
			size_t prev_layer_id = layer_id - 1;
			
			for (size_t k = 0; k < net.weights[prev_layer_id].ncols; ++k) {
			  float prev_weight = MAT_GET(net.weights[prev_layer_id], j, k);
			  MAT_GET(net.weights[prev_layer_id],j, k) -= change * MAT_GET(net.activations[prev_layer_id], 0,k);
			  
			  prev_act_change = MAT_GET(net.weights[prev_layer_id], j, k) * change;
			  MAT_GET(net.activations[prev_layer_id], 0, k) -= prev_weight * change;
			}
			MAT_GET(net.biases[prev_layer_id], 0, j) -= change;
        	}
        }
    }
}

#endif // CLEAR_NET_IMPLEMENTATION
