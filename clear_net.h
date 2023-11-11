/***
    Clear Net (c) by Timothy Cronin

    Clear Net is licensed under a
    Creative Commons Attribution 4.0 International License.

    You should have received a copy of the license along with this
    work.  If not, see <http://creativecommons.org/licenses/by/4.0/>.

    License at bottom of file.
***/

/***
    TODO do all computations on the computation graph, no more of just vanilla
float implementation, need to output the floats so have some function which
takes the gs id to floats, each net has a matrix/vector for output float values,
after the final layer change the output_ids to just the output
matrix/matrices/vector
    TODO come up with some way to use the typedef for scalar and not the float
specifics like powf
    TODO instead of funciton pointers do operation type enum
    TODO way to write the forwarding once for both vanilla floats and storing
computation graph
    TODO parallel
    TODO more loss functions,
    TODO more optimization methods
    TODO do a segmentation example with matrix output
***/

/* Beginning */
#ifndef CLEAR_NET
#define CLEAR_NET

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// allow custom memory allocation strategies
#ifndef CLEAR_NET_ALLOC
#define CLEAR_NET_ALLOC malloc
#endif // CLEAR_NET_ALLOC
#ifndef CLEAR_NET_REALLOC
#define CLEAR_NET_REALLOC realloc
#endif // CLEAR_NET_REALLOC
#ifndef CLEAR_NET_DEALLOC
#define CLEAR_NET_DEALLOC free
#endif // CLEAR_NET_MALLOC
// allow custom assertion strategies
#ifndef CLEAR_NET_ASSERT
#include "assert.h"
#define CLEAR_NET_ASSERT assert
#endif // CLEAR_NET_ASSERT
#ifndef CLEAR_NET_INITIAL_GRAPH_LENGTH
#define CLEAR_NET_INITIAL_GRAPH_LENGTH 10
#endif // CLEAR_NET_INITIAL_GRAPH_LENGTH
#ifndef CLEAR_NET_DEFINE_HYPERPARAMETERS
#define CLEAR_NET_DEFINE_HYPERPARAMETERS                                       \
    NetType CN_NET_TYPE;                                                       \
    scalar CN_RATE;                                                            \
    size_t CN_NLAYERS;                                                         \
    size_t CN_NPARAMS;                                                         \
    scalar CN_NEG_SCALE;                                                       \
    size_t CN_WITH_MOMENTUM;                                                   \
    scalar CN_MOMENTUM_BETA;
#endif // CLEAR_NET_DEFINE_HYPERPARAMETERS
#ifndef CN_MAX_PARAM
#define CN_MAX_PARAM FLT_MAX
#endif // CN_MAX_PARAM

typedef float scalar;

/* Declare: Helpers */
void _cn_fill_params(scalar *ptr, size_t len, scalar val);

/* Declare: Hyper Parameters */
void cn_default_hparams(void);
void cn_with_momentum(scalar momentum_beta);
void cn_set_neg_scale(scalar neg_scale);
void cn_set_rate(scalar rate);

/* Declare: Activation Functions */
typedef enum {
    Sigmoid,
    ReLU,
    Tanh,
    LeakyReLU,
    ELU,
} Activation;

scalar cn_sigmoid(scalar x);
scalar cn_relu(scalar x);
scalar cn_hyper_tan(scalar x);
scalar cn_leaky_relu(scalar x);
scalar cn_elu(scalar x);
scalar cn_activate(scalar x, Activation act);

/* Declare: Automatic Differentiation Engine */
typedef struct VarNode VarNode;
typedef struct GradientStore GradientStore;
typedef void BackWardFunction(GradientStore *nl, VarNode *var);

GradientStore cn_alloc_gradient_store(size_t max_length);
void cn_realloc_gradient_store(GradientStore *gs, size_t new_len);
void cn_dealloc_gradient_store(GradientStore *nl);
VarNode _cn_create_var(scalar num, size_t prev_left, size_t prev_right,
                       BackWardFunction *backward);
size_t _cn_init_var(GradientStore *nl, scalar num, size_t prev_left,
                    size_t prev_right, BackWardFunction *backward);
size_t cn_init_leaf_var(GradientStore *nl, scalar num);
void _cn_add_backward(GradientStore *nl, VarNode *var);
size_t cn_add(GradientStore *nl, size_t left, size_t right);
void _cn_subtract_backward(GradientStore *nl, VarNode *var);
size_t cn_subtract(GradientStore *nl, size_t left, size_t right);
void _cn_multiply_backward(GradientStore *nl, VarNode *var);
size_t cn_multiply(GradientStore *nl, size_t left, size_t right);
void _cn_raise_backward(GradientStore *gs, VarNode *var);
size_t cn_raise(GradientStore *gs, size_t to_raise, size_t pow);
void _cn_relu_backward(GradientStore *nl, VarNode *var);
size_t cn_reluv(GradientStore *nl, size_t x);
void _cn_tanh_backward(GradientStore *nl, VarNode *var);
size_t cn_hyper_tanv(GradientStore *nl, size_t x);
void _cn_sigmoid_backward(GradientStore *nl, VarNode *var);
size_t cn_sigmoidv(GradientStore *nl, size_t x);
void _cn_leaky_relu_backward(GradientStore *nl, VarNode *var);
size_t cn_leaky_reluv(GradientStore *nl, size_t x);
void _cn_elu_backward(GradientStore *gs, VarNode *var);
size_t cn_eluv(GradientStore *gs, size_t x);
size_t _cn_activate(GradientStore *nl, size_t id, Activation act);
void cn_backward(GradientStore *nl, size_t y);

/* Declare: Linear Algebra */
typedef struct Matrix Matrix;
typedef struct DMatrix DMatrix;
Matrix cn_alloc_matrix(size_t nrows, size_t ncols);
void cn_dealloc_matrix(Matrix *mat);
void _cn_alloc_matrix_grad_stores(Matrix *mat);
DMatrix _cn_alloc_dmatrix(size_t nrows, size_t ncols);
void _cn_dealloc_dmatrix(DMatrix *mat);
Matrix cn_form_matrix(size_t nrows, size_t ncols, size_t stride,
                      scalar *elements);
void _cn_randomize_matrix(Matrix *mat, scalar lower, scalar upper);
void _cn_copy_matrix_params(GradientStore *gs, Matrix mat);
void _cn_apply_matrix_grads(GradientStore *gs, Matrix *mat);
void cn_print_matrix(Matrix mat, char *name);

typedef struct Vector Vector;
typedef struct DVector DVector;
Vector cn_alloc_vector(size_t nelem);
void cn_dealloc_vector(Vector *vec);
void _cn_alloc_vector_grad_stores(Vector *vec);
DVector _cn_alloc_dvector(size_t nelem);
void _cn_dealloc_dvector(DVector *vec);
Vector cn_form_vector(size_t nelem, scalar *elements);
void _cn_randomize_vector(Vector *vec, scalar lower, scalar upper);
void _cn_copy_vector_params(GradientStore *gs, Vector vec);
void _cn_apply_vector_grads(GradientStore *gs, Vector *vec);
void _cn_print_vector(Vector vec, char *name);
void cn_print_vector_inline(Vector vec);

typedef enum {
    Vec,
    Mat,
    MatList,
} LAType;
typedef union VecMat VecMat;
typedef struct LAData LAData;
typedef union DVecMat DVecMat;
typedef struct DLAData DLAData;

/* Declare: Net Structs */
typedef struct DenseLayer DenseLayer;
typedef struct ConvolutionalLayer ConvolutionalLayer;
typedef struct Filter Filter;
typedef enum {
    Same,
    Valid,
    Full,
} Padding;
typedef struct PoolingLayer PoolingLayer;
typedef struct GlobalPoolingLayer GlobalPoolingLayer;
typedef enum {
    Max,
    Average,
} PoolingStrategy;
typedef union LayerData LayerData;
typedef enum {
    Dense,
    Conv,
    Pooling,
    GlobalPooling,
} LayerType;
typedef struct Layer Layer;
typedef struct Net Net;
typedef enum {
    Vanilla,
    Convolutional,
} NetType;

/* Declare: DenseLayer */
void cn_alloc_dense_layer(Net *net, Activation act, size_t dim_output);
void _cn_dealloc_dense_layer(DenseLayer *layer);
void _cn_randomize_dense_layer(DenseLayer *layer, scalar lower, scalar upper);
void _cn_copy_dense_params(GradientStore *gs, DenseLayer *layer);
Vector cn_forward_dense(DenseLayer *layer, Vector prev_output);
DVector _cn_forward_dense(DenseLayer *layer, GradientStore *gs,
                          DVector prev_ids);
void _cn_apply_dense_grads(GradientStore *gs, DenseLayer *layer);
void _cn_save_dense_layer_to_file(FILE *fp, DenseLayer *layer);
void _cn_alloc_dense_layer_from_file(FILE *fp, Net *net);
void _cn_print_dense(DenseLayer dense, size_t index);

/* Declare: Convolutional Layer */
void cn_alloc_conv_layer(Net *net, Padding padding, Activation act,
                         size_t noutput, size_t kernel_nrows,
                         size_t kernel_ncols);
void _cn_set_padding(Padding padding, size_t knrows, size_t kncols,
                     size_t *row_padding, size_t *col_padding);
void _cn_dealloc_conv_layer(ConvolutionalLayer *layer);
void _cn_randomize_conv_layer(ConvolutionalLayer *layer, scalar lower,
                              scalar upper);
void _cn_copy_conv_params(GradientStore *gs, ConvolutionalLayer *layer);
Matrix *cn_forward_conv(ConvolutionalLayer *layer, Matrix *input);
scalar cn_correlate(Matrix kern, Matrix input, long top_left_row,
                    long top_left_col);
size_t _cn_correlate(GradientStore *gs, Matrix kern, DMatrix input,
                     long top_left_row, long top_left_col);
DMatrix *_cn_forward_conv(ConvolutionalLayer *layer, GradientStore *gs,
                          DMatrix *input);
void _cn_apply_conv_grads(GradientStore *gs, ConvolutionalLayer *layer);
void _cn_save_conv_layer_to_file(FILE *fp, ConvolutionalLayer *layer);
void _cn_alloc_conv_layer_from_file(FILE *fp, Net *net);
void _cn_print_conv_layer(ConvolutionalLayer *layer, size_t index);

/* Declare: Pooling Layer */
void cn_alloc_pooling_layer(Net *net, PoolingStrategy strat,
                            size_t kernel_nrows, size_t kernel_ncols);
Matrix *cn_pool_layer(PoolingLayer *pooler, Matrix *input);
DMatrix *_cn_pool_layer(GradientStore *gs, PoolingLayer *pooler,
                        DMatrix *input);
void _cn_save_pooling_layer_to_file(FILE *fp, PoolingLayer *layer);
void _cn_alloc_pooling_layer_from_file(FILE *fp, Net *net);
void _cn_print_pooling_layer(PoolingLayer layer, size_t layer_id);
void _cn_dealloc_pooling_layer(PoolingLayer *layer);

/* Declare: Global Pooling Layer */
void cn_alloc_global_pooling_layer(Net *net, PoolingStrategy strat);
Vector cn_global_pool_layer(GlobalPoolingLayer *pooler, Matrix *input);
DVector _cn_global_pool_layer(GradientStore *gs, GlobalPoolingLayer *pooler,
                              DMatrix *input);
void _cn_save_global_pooling_layer_to_file(FILE *fp, GlobalPoolingLayer *layer);
void _cn_alloc_global_pooling_layer_from_file(FILE *fp, Net *net);
void _cn_print_global_pooling_layer(GlobalPoolingLayer gpooler,
                                    size_t layer_id);
void _cn_dealloc_global_pooling_layer(GlobalPoolingLayer *layer);

/* Declare: Layer */
Layer _cn_init_layer(LayerType type);

/* Declare: Net */
Net cn_alloc_conv_net(size_t input_nrows, size_t input_ncols, size_t nchannels);
Net cn_alloc_vani_net(size_t input_nelem);
void cn_dealloc_net(Net *net);
void _cn_add_net_layer(Net *net, Layer layer, size_t nparams);
void _cn_copy_net_params(Net *net, GradientStore *gs);
void cn_randomize_net(Net *net, scalar lower, scalar upper);
void cn_shuffle_vani_input(Matrix *input, Matrix *target);
void cn_get_batch_vani(Matrix *batch_in, Matrix *batch_tar, Matrix all_input,
                       Matrix all_target, size_t batch_num, size_t batch_size);
void cn_shuffle_conv_input(Matrix ***input, LAData **targets, size_t len);
void cn_get_batch_conv(Matrix **batch_in, LAData *batch_tar, Matrix **all_input,
                       LAData *all_target, size_t batch_num, size_t batch_size);
void cn_save_net_to_file(Net net, char *file_name);
Net cn_alloc_net_from_file(char *file_name);
void cn_print_net(Net net, char *name);

/* Declare: Vanilla */
scalar cn_learn_vani(Net *net, Matrix input, Matrix target);
scalar _cn_find_grad_vani(Net *net, GradientStore *gs, Vector input,
                          Vector target);
Vector cn_predict_vani(Net *net, Vector input);
DVector _cn_predict_vani(Net *net, GradientStore *gs, DVector prev_ids);
scalar cn_loss_vani(Net *net, Matrix input, Matrix target);
void cn_print_vani_results(Net net, Matrix input, Matrix target);
void cn_print_target_output_pairs_vani(Net net, Matrix input, Matrix target);

/* Declare: Convolutional Net */
scalar cn_learn_conv(Net *net, Matrix **inputs, LAData *targets, size_t nimput);
scalar _cn_find_grad_conv(Net *net, GradientStore *gs, Matrix *inputs,
                          LAData target);
LAData cn_predict_conv(Net *net, Matrix *input);
DLAData _cn_predict_conv(Net *net, GradientStore *gs, DMatrix *minput);
scalar cn_loss_conv(Net *net, Matrix **input, LAData *targets, size_t nimput);
void cn_print_conv_results(Net net, Matrix **inputs, LAData *targets,
                           size_t nimput);

#endif // CLEAR_NET

#ifdef CLEAR_NET_IMPLEMENTATION

/* Implement: Helpers */
void _cn_fill_params(scalar *ptr, size_t len, scalar val) {
    for (size_t i = 0; i < len; ++i) {
        ptr[i] = val;
    }
}

#define RAND_RANGE(upper, lower)                                               \
    (((scalar)rand() / RAND_MAX) * ((upper) - (lower)) + (lower))

/* Implement: Hyper parameters */
extern NetType CN_NET_TYPE;
extern scalar CN_RATE;
extern size_t CN_NLAYERS;
extern size_t CN_NPARAMS;
extern scalar CN_NEG_SCALE;
extern size_t CN_WITH_MOMENTUM;
extern scalar CN_MOMENTUM_BETA;

void cn_default_hparams(void) {
    CN_RATE = 0.01;
    CN_NLAYERS = 0;
    CN_NPARAMS = 0;
    CN_NEG_SCALE = 0.1;
    CN_WITH_MOMENTUM = 0;
    CN_MOMENTUM_BETA = 0;
}

void cn_with_momentum(scalar momentum_beta) {
    CN_WITH_MOMENTUM = 1;
    CN_MOMENTUM_BETA = momentum_beta;
}
void cn_set_neg_scale(scalar neg_scale) { CN_NEG_SCALE = neg_scale; }
void cn_set_rate(scalar rate) { CN_RATE = rate; }

/* Implement: Activation Functions */
scalar cn_sigmoid(scalar x) { return 1 / (1 + expf(-x)); }
scalar cn_relu(scalar x) { return x > 0 ? x : 0; }
scalar cn_hyper_tan(scalar x) { return tanhf(x); }
scalar cn_leaky_relu(scalar x) { return x >= 0 ? x : CN_NEG_SCALE * x; }
scalar cn_elu(scalar x) { return x > 0 ? x : CN_NEG_SCALE * (expf(x) - 1); }
scalar cn_activate(scalar x, Activation act) {
    switch (act) {
    case ReLU:
        return cn_relu(x);
    case Sigmoid:
        return cn_sigmoid(x);
    case Tanh:
        return cn_hyper_tan(x);
    case LeakyReLU:
        return cn_leaky_relu(x);
    case ELU:
        return cn_elu(x);
    }
}

/* Implement: Automatic Differentiation Engine */
#define CLEAR_NET_EXTEND_LENGTH_FUNCTION(len)                                  \
    ((len) == 0 ? CLEAR_NET_INITIAL_GRAPH_LENGTH : ((len)*2))
#define GET_NODE(id) (gs)->vars[(id)]

struct VarNode {
    scalar num;
    scalar grad;
    BackWardFunction *backward;
    size_t prev_left;
    size_t prev_right;
};

struct GradientStore {
    VarNode *vars;
    size_t length;
    size_t max_length;
};

GradientStore cn_alloc_gradient_store(size_t max_length) {
    return (GradientStore){
        .vars = CLEAR_NET_ALLOC(max_length * sizeof(VarNode)),
        .length = 1,
        .max_length = max_length,
    };
}

void cn_realloc_gradient_store(GradientStore *gs, size_t new_len) {
    gs->vars = CLEAR_NET_REALLOC(gs->vars, new_len * sizeof(*gs->vars));
    gs->max_length = new_len;
}

void cn_dealloc_gradient_store(GradientStore *gs) {
    CLEAR_NET_DEALLOC(gs->vars);
}

VarNode _cn_create_var(scalar num, size_t prev_left, size_t prev_right,
                       BackWardFunction *backward) {
    return (VarNode){
        .num = num,
        .grad = 0,
        .prev_left = prev_left,
        .prev_right = prev_right,
        .backward = backward,
    };
}

size_t _cn_init_var(GradientStore *gs, scalar num, size_t prev_left,
                    size_t prev_right, BackWardFunction *backward) {
    if (gs->length >= gs->max_length) {
        gs->max_length = CLEAR_NET_EXTEND_LENGTH_FUNCTION(gs->max_length);
        gs->vars =
            CLEAR_NET_REALLOC(gs->vars, gs->max_length * sizeof(VarNode));
        CLEAR_NET_ASSERT(gs->vars);
    }
    VarNode out = _cn_create_var(num, prev_left, prev_right, backward);
    gs->vars[gs->length] = out;
    gs->length++;
    return gs->length - 1;
}

size_t cn_init_leaf_var(GradientStore *gs, scalar num) {
    return _cn_init_var(gs, num, 0, 0, NULL);
}

void _cn_add_backward(GradientStore *gs, VarNode *var) {
    GET_NODE(var->prev_left).grad += var->grad;
    GET_NODE(var->prev_right).grad += var->grad;
}

size_t cn_add(GradientStore *gs, size_t left, size_t right) {
    scalar val = GET_NODE(left).num + GET_NODE(right).num;
    size_t out = _cn_init_var(gs, val, left, right, _cn_add_backward);
    return out;
}

void _cn_subtract_backward(GradientStore *gs, VarNode *var) {
    GET_NODE(var->prev_left).grad += var->grad;
    GET_NODE(var->prev_right).grad -= var->grad;
}

size_t cn_subtract(GradientStore *gs, size_t left, size_t right) {
    scalar val = GET_NODE(left).num - GET_NODE(right).num;
    size_t out = _cn_init_var(gs, val, left, right, _cn_subtract_backward);
    return out;
}

void _cn_multiply_backward(GradientStore *gs, VarNode *var) {
    GET_NODE(var->prev_left).grad += GET_NODE(var->prev_right).num * var->grad;
    GET_NODE(var->prev_right).grad += GET_NODE(var->prev_left).num * var->grad;
}

size_t cn_multiply(GradientStore *gs, size_t left, size_t right) {
    scalar val = GET_NODE(left).num * GET_NODE(right).num;
    size_t out = _cn_init_var(gs, val, left, right, _cn_multiply_backward);
    return out;
}

void _cn_raise_backward(GradientStore *gs, VarNode *var) {
    scalar l_num = GET_NODE(var->prev_left).num;
    scalar r_num = GET_NODE(var->prev_right).num;
    GET_NODE(var->prev_left).grad += r_num * powf(l_num, r_num - 1) * var->grad;
    GET_NODE(var->prev_right).grad +=
        logf(l_num) * powf(l_num, r_num) * var->grad;
}

size_t cn_raise(GradientStore *gs, size_t to_raise, size_t pow) {
    scalar val = powf(GET_NODE(to_raise).num, GET_NODE(pow).num);
    size_t out = _cn_init_var(gs, val, to_raise, pow, _cn_raise_backward);
    return out;
}

void _cn_relu_backward(GradientStore *gs, VarNode *var) {
    if (var->num > 0) {
        GET_NODE(var->prev_left).grad += var->grad;
    }
}

size_t cn_reluv(GradientStore *gs, size_t x) {
    scalar val = cn_relu(GET_NODE(x).num);
    size_t out = _cn_init_var(gs, val, x, 0, _cn_relu_backward);
    return out;
}

void _cn_tanh_backward(GradientStore *gs, VarNode *var) {
    GET_NODE(var->prev_left).grad += (1 - powf(var->num, 2)) * var->grad;
}

size_t cn_hyper_tanv(GradientStore *gs, size_t x) {
    scalar val = cn_hyper_tan(GET_NODE(x).num);
    size_t out = _cn_init_var(gs, val, x, 0, _cn_tanh_backward);
    return out;
}

void _cn_sigmoid_backward(GradientStore *gs, VarNode *var) {
    GET_NODE(var->prev_left).grad += var->num * (1 - var->num) * var->grad;
}

size_t cn_sigmoidv(GradientStore *gs, size_t x) {
    scalar val = cn_sigmoid(GET_NODE(x).num);
    size_t out = _cn_init_var(gs, val, x, 0, _cn_sigmoid_backward);
    return out;
}

void _cn_leaky_relu_backward(GradientStore *gs, VarNode *var) {
    scalar change = var->num >= 0 ? 1 : CN_NEG_SCALE;
    GET_NODE(var->prev_left).grad += change * var->grad;
}

size_t cn_leaky_reluv(GradientStore *gs, size_t x) {
    scalar val = cn_leaky_relu(GET_NODE(x).num);
    size_t out = _cn_init_var(gs, val, x, 0, _cn_leaky_relu_backward);
    return out;
}

void _cn_elu_backward(GradientStore *gs, VarNode *var) {
    scalar change = var->num > 0 ? 1 : var->num + CN_NEG_SCALE;
    GET_NODE(var->prev_left).grad += change * var->grad;
}

size_t cn_eluv(GradientStore *gs, size_t x) {
    scalar val = cn_elu(GET_NODE(x).num);
    size_t out = _cn_init_var(gs, val, x, 0, _cn_elu_backward);
    return out;
}

size_t _cn_activate(GradientStore *gs, size_t id, Activation act) {
    switch (act) {
    case ReLU:
        return cn_reluv(gs, id);
    case Sigmoid:
        return cn_sigmoidv(gs, id);
    case Tanh:
        return cn_hyper_tanv(gs, id);
    case LeakyReLU:
        return cn_leaky_reluv(gs, id);
    case ELU:
        return cn_eluv(gs, id);
    }
}

void cn_backward(GradientStore *gs, size_t y) {
    GET_NODE(y).grad = 1;
    VarNode *var;
    for (size_t i = gs->length - 1; i > 0; --i) {
        var = &GET_NODE(i);
        if (var->backward) {
            var->backward(gs, var);
        }
    }
}

/* Implement: Linear Algebra */
#define MAT_ID(mat, r, c) (mat).gs_id + ((r) * (mat).stride) + (c)
#define MAT_AT(mat, r, c) (mat).elements[(r) * (mat).stride + (c)]
#define VEC_ID(vec, i) (vec).gs_id + (i)
#define VEC_AT(vec, i) (vec).elements[i]
#define CN_PRINT_MATRIX(mat) cn_print_matrix((mat), #mat)
#define _CN_PRINT_VECTOR(vec) _cn_print_vector((vec), #vec)

struct Matrix {
    scalar *elements;
    scalar *grad_stores;
    size_t gs_id;
    size_t stride;
    size_t nrows;
    size_t ncols;
};

struct DMatrix {
    size_t *elements;
    size_t stride;
    size_t nrows;
    size_t ncols;
};

Matrix cn_alloc_matrix(size_t nrows, size_t ncols) {
    Matrix mat;
    mat.nrows = nrows;
    mat.ncols = ncols;
    mat.stride = ncols;
    mat.elements = CLEAR_NET_ALLOC(nrows * ncols * sizeof(*mat.elements));
    CLEAR_NET_ASSERT(mat.elements != NULL);
    mat.grad_stores = NULL;
    mat.gs_id = 0;
    return mat;
}

void cn_dealloc_matrix(Matrix *mat) {
    CLEAR_NET_DEALLOC(mat->elements);
    mat->elements = NULL;
    if (mat->grad_stores != NULL) {
        CLEAR_NET_DEALLOC(mat->grad_stores);
    }
    mat->nrows = 0;
    mat->ncols = 0;
    mat->stride = 0;
    mat->gs_id = 0;
}

void _cn_alloc_matrix_grad_stores(Matrix *mat) {
    mat->grad_stores =
        CLEAR_NET_ALLOC(mat->nrows * mat->ncols * sizeof(*mat->grad_stores));
    _cn_fill_params(mat->grad_stores, mat->nrows * mat->ncols, 0);
}

DMatrix _cn_alloc_dmatrix(size_t nrows, size_t ncols) {
    DMatrix mat;
    mat.nrows = nrows;
    mat.ncols = ncols;
    mat.stride = ncols;
    mat.elements = CLEAR_NET_ALLOC(nrows * ncols * sizeof(*mat.elements));
    CLEAR_NET_ASSERT(mat.elements != NULL);
    return mat;
}

void _cn_dealloc_dmatrix(DMatrix *mat) {
    CLEAR_NET_DEALLOC(mat->elements);
    mat->elements = NULL;
    mat->nrows = 0;
    mat->ncols = 0;
    mat->stride = 0;
}

Matrix cn_form_matrix(size_t nrows, size_t ncols, size_t stride,
                      scalar *elements) {
    return (Matrix){.gs_id = 0,
                    .nrows = nrows,
                    .ncols = ncols,
                    .stride = stride,
                    .elements = elements};
}

void _cn_randomize_matrix(Matrix *mat, scalar lower, scalar upper) {
    for (size_t j = 0; j < mat->nrows; ++j) {
        for (size_t k = 0; k < mat->ncols; ++k) {
            MAT_AT(*mat, j, k) = RAND_RANGE(lower, upper);
        }
    }
}

void _cn_copy_matrix_params(GradientStore *gs, Matrix mat) {
    for (size_t i = 0; i < mat.nrows; ++i) {
        for (size_t j = 0; j < mat.ncols; ++j) {
            cn_init_leaf_var(gs, MAT_AT(mat, i, j));
        }
    }
}

void _cn_apply_matrix_grads(GradientStore *gs, Matrix *mat) {
    for (size_t i = 0; i < mat->nrows; ++i) {
        for (size_t j = 0; j < mat->ncols; ++j) {
            scalar change = CN_RATE * GET_NODE(MAT_ID(*mat, i, j)).grad;
            if (CN_WITH_MOMENTUM) {
                size_t id = (i * mat->stride) + j;
                mat->grad_stores[id] = CN_MOMENTUM_BETA * mat->grad_stores[id] +
                                       ((1 - CN_MOMENTUM_BETA) * change);
                change = mat->grad_stores[id];
            }
            MAT_AT(*mat, i, j) -= change;
        }
    }
}

void cn_print_matrix(Matrix mat, char *name) {
    printf("%s = [\n", name);
    for (size_t i = 0; i < mat.nrows; ++i) {
        printf("    ");
        for (size_t j = 0; j < mat.ncols; ++j) {
            printf("%f ", MAT_AT(mat, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}

struct Vector {
    scalar *elements;
    scalar *grad_stores;
    size_t gs_id;
    size_t nelem;
};

struct DVector {
    size_t *elements;
    size_t nelem;
};

Vector cn_alloc_vector(size_t nelem) {
    Vector vec;
    vec.nelem = nelem;
    vec.elements = CLEAR_NET_ALLOC(nelem * sizeof(*vec.elements));
    CLEAR_NET_ASSERT(vec.elements != NULL);
    vec.grad_stores = NULL;
    return vec;
}

void cn_dealloc_vector(Vector *vec) {
    CLEAR_NET_DEALLOC(vec->elements);
    if (vec->grad_stores != NULL) {
        CLEAR_NET_DEALLOC(vec->grad_stores);
    }
    vec->nelem = 0;
    vec->gs_id = 0;
    vec->elements = NULL;
}

void _cn_alloc_vector_grad_stores(Vector *vec) {
    vec->grad_stores = CLEAR_NET_ALLOC(vec->nelem * sizeof(*vec->grad_stores));
    _cn_fill_params(vec->grad_stores, vec->nelem, 0);
}

DVector _cn_alloc_dvector(size_t nelem) {
    DVector vec;
    vec.nelem = nelem;
    vec.elements = CLEAR_NET_ALLOC(nelem * sizeof(*vec.elements));
    CLEAR_NET_ASSERT(vec.elements != NULL);
    return vec;
}

void _cn_dealloc_dvector(DVector *vec) {
    CLEAR_NET_DEALLOC(vec->elements);
    vec->nelem = 0;
    vec->elements = NULL;
}

Vector cn_form_vector(size_t nelem, scalar *elements) {
    return (Vector){
        .gs_id = 0,
        .nelem = nelem,
        .elements = elements,
    };
}

void _cn_randomize_vector(Vector *vec, scalar lower, scalar upper) {
    for (size_t i = 0; i < vec->nelem; ++i) {
        VEC_AT(*vec, i) = RAND_RANGE(lower, upper);
    }
}

void _cn_copy_vector_params(GradientStore *gs, Vector vec) {
    for (size_t i = 0; i < vec.nelem; ++i) {
        cn_init_leaf_var(gs, VEC_AT(vec, i));
    }
}

void _cn_apply_vector_grads(GradientStore *gs, Vector *vec) {
    for (size_t i = 0; i < vec->nelem; ++i) {
        scalar change = CN_RATE * GET_NODE(VEC_ID(*vec, i)).grad;
        if (CN_WITH_MOMENTUM) {
            vec->grad_stores[i] = CN_MOMENTUM_BETA * vec->grad_stores[i] +
                                  ((1 - CN_MOMENTUM_BETA) * change);
            change = vec->grad_stores[i];
        }
        VEC_AT(*vec, i) -= change;
    }
}

void _cn_print_vector(Vector vec, char *name) {
    printf("%s = [\n", name);
    printf("    ");
    for (size_t i = 0; i < vec.nelem; ++i) {
        printf("%f ", VEC_AT(vec, i));
    }
    printf("\n]\n");
}

void cn_print_vector_inline(Vector vec) {
    for (size_t j = 0; j < vec.nelem; ++j) {
        printf("%f ", VEC_AT(vec, j));
    }
}

/* Implement Vec+Mat Wrapper */
union VecMat {
    Vector vec;
    Matrix mat;
    Matrix *mat_list;
};

struct LAData {
    VecMat data;
    LAType type;
};

union DVecMat {
    DVector vec;
    DMatrix mat;
    DMatrix *mat_list;
};

struct DLAData {
    DVecMat data;
    size_t nimput;
    LAType type;
};

void _cn_dealloc_dladata(DLAData *data) {
    switch (data->type) {
    case Vec:
        _cn_dealloc_dvector(&data->data.vec);
        break;
    case Mat:
        _cn_dealloc_dmatrix(&data->data.mat);
        break;
    case MatList:
        for (size_t i = 0; i < data->nimput; ++i) {
            _cn_dealloc_dmatrix(&data->data.mat_list[i]);
        }
        CLEAR_NET_DEALLOC(data->data.mat_list);
        break;
    }
    data->nimput = 0;
    data->type = 0;
}

/* Implement: Net Structs */
struct DenseLayer {
    Matrix weights;
    Vector biases;
    Activation act;
    DVector output_ids;
    Vector output;
};

struct ConvolutionalLayer {
    Filter *filters;
    Matrix *outputs;
    DMatrix *output_ids;
    size_t nfilters;
    Padding padding;
    Activation act;
    size_t nimput;
    size_t input_nrows;
    size_t input_ncols;
    size_t output_nrows;
    size_t output_ncols;
    size_t k_nrows;
    size_t k_ncols;
};

struct Filter {
    Matrix *kernels;
    Matrix biases;
};

struct PoolingLayer {
    Matrix *outputs;
    DMatrix *output_ids;
    PoolingStrategy strat;
    size_t noutput;
    size_t k_nrows;
    size_t k_ncols;
    size_t output_nrows;
    size_t output_ncols;
};

struct GlobalPoolingLayer {
    Vector output;
    DVector output_ids;
    PoolingStrategy strat;
};

struct Net {
    DLAData input_ids;
    Layer *layers;
    GradientStore computation_graph;
    LAType output_type;
};

union LayerData {
    DenseLayer dense;
    ConvolutionalLayer conv;
    PoolingLayer pooling;
    GlobalPoolingLayer global_pooling;
};

struct Layer {
    LayerType type;
    LayerData data;
};

/* Implement: DenseLayer */
void cn_alloc_dense_layer(Net *net, Activation act, size_t dim_output) {
    size_t dim_input;
    if (CN_NLAYERS != 0) {
        Layer player = net->layers[CN_NLAYERS - 1];
        CLEAR_NET_ASSERT(player.type == Dense || player.type == GlobalPooling);
        if (player.type == Dense) {
            dim_input = player.data.dense.output.nelem;
        }
        if (player.type == GlobalPooling) {
            dim_input = player.data.global_pooling.output.nelem;
        }
    } else {
        dim_input = net->input_ids.data.vec.nelem;
    }
    DenseLayer dense_layer;
    dense_layer.act = act;

    size_t offset = CN_NPARAMS + 1;
    Matrix weights = cn_alloc_matrix(dim_input, dim_output);
    if (CN_WITH_MOMENTUM) {
        _cn_alloc_matrix_grad_stores(&weights);
    }
    weights.gs_id = offset;
    dense_layer.weights = weights;
    offset += dense_layer.weights.nrows * dense_layer.weights.ncols;

    Vector biases = cn_alloc_vector(dim_output);
    biases.gs_id = offset;
    if (CN_WITH_MOMENTUM) {
        _cn_alloc_vector_grad_stores(&biases);
    }
    dense_layer.biases = biases;
    offset += dense_layer.biases.nelem;

    Vector out = cn_alloc_vector(dim_output);
    out.gs_id = 0;
    dense_layer.output = out;
    dense_layer.output_ids = _cn_alloc_dvector(dense_layer.biases.nelem);

    CN_NPARAMS = offset - 1;
    Layer layer = _cn_init_layer(Dense);
    layer.data.dense = dense_layer;

    _cn_add_net_layer(net, layer, CN_NPARAMS);
    net->output_type = Vec;
}

void _cn_set_padding(Padding padding, size_t knrows, size_t kncols,
                     size_t *row_padding, size_t *col_padding) {
    switch (padding) {
    case Same:
        *row_padding = (knrows - 1) / 2;
        *col_padding = (kncols - 1) / 2;
        break;
    case Full:
        *row_padding = knrows - 1;
        *col_padding = kncols - 1;
        break;
    case Valid:
        *row_padding = 0;
        *col_padding = 0;
        break;
    }
}

void _cn_dealloc_dense_layer(DenseLayer *layer) {
    cn_dealloc_matrix(&layer->weights);
    cn_dealloc_vector(&layer->biases);
    cn_dealloc_vector(&layer->output);
    _cn_dealloc_dvector(&layer->output_ids);
}

void _cn_randomize_dense_layer(DenseLayer *layer, scalar lower, scalar upper) {
    _cn_randomize_matrix(&layer->weights, lower, upper);
    _cn_randomize_vector(&layer->biases, lower, upper);
}

void _cn_copy_dense_params(GradientStore *gs, DenseLayer *layer) {
    _cn_copy_matrix_params(gs, layer->weights);
    _cn_copy_vector_params(gs, layer->biases);
}

Vector cn_forward_dense(DenseLayer *layer, Vector prev_output) {
    for (size_t i = 0; i < layer->weights.ncols; ++i) {
        scalar res = 0;
        for (size_t j = 0; j < prev_output.nelem; ++j) {
            res += MAT_AT(layer->weights, j, i) * VEC_AT(prev_output, j);
        }
        res += VEC_AT(layer->biases, i);
        res = cn_activate(res, layer->act);
        layer->output.elements[i] = res;
    }
    return layer->output;
}

DVector _cn_forward_dense(DenseLayer *layer, GradientStore *gs,
                          DVector prev_ids) {
    for (size_t i = 0; i < layer->weights.ncols; ++i) {
        size_t res = cn_init_leaf_var(gs, 0);
        for (size_t j = 0; j < prev_ids.nelem; ++j) {
            res = cn_add(gs, res,
                         cn_multiply(gs, MAT_ID(layer->weights, j, i),
                                     VEC_AT(prev_ids, j)));
        }
        res = cn_add(gs, res, VEC_ID(layer->biases, i));
        res = _cn_activate(gs, res, layer->act);
        VEC_AT(layer->output_ids, i) = res;
    }

    return layer->output_ids;
}

void _cn_apply_dense_grads(GradientStore *gs, DenseLayer *layer) {
    _cn_apply_matrix_grads(gs, &layer->weights);
    _cn_apply_vector_grads(gs, &layer->biases);
}

void _cn_save_dense_layer_to_file(FILE *fp, DenseLayer *layer) {
    fwrite(&layer->act, sizeof(layer->act), 1, fp);
    Matrix weights = layer->weights;
    Vector biases = layer->biases;
    fwrite(&weights.ncols, sizeof(weights.ncols), 1, fp);
    fwrite(weights.elements, sizeof(*weights.elements),
           weights.nrows * weights.ncols, fp);
    fwrite(biases.elements, sizeof(*biases.elements), biases.nelem, fp);
}

void _cn_alloc_dense_layer_from_file(FILE *fp, Net *net) {

    Activation act;
    fread(&act, sizeof(act), 1, fp);
    size_t output_dim;
    fread(&output_dim, sizeof(output_dim), 1, fp);
    cn_alloc_dense_layer(net, act, output_dim);

    Matrix weights = net->layers[CN_NLAYERS - 1].data.dense.weights;
    fread(weights.elements, sizeof(*weights.elements),
          weights.nrows * weights.ncols, fp);
    Vector biases = net->layers[CN_NLAYERS - 1].data.dense.biases;
    fread(biases.elements, sizeof(*biases.elements), biases.nelem, fp);
}

void _cn_print_dense(DenseLayer dense, size_t index) {
    printf("Layer #%zu: Dense\n", index);
    cn_print_matrix(dense.weights, "weight matrix");
    _cn_print_vector(dense.biases, "bias vector");
}

/* Implement: Convolutional Layer */
void cn_alloc_conv_layer(Net *net, Padding padding, Activation act,
                         size_t noutput, size_t kernel_nrows,
                         size_t kernel_ncols) {
    size_t input_nrows;
    size_t input_ncols;
    size_t nimput;
    if (CN_NLAYERS != 0) {
        Layer player = net->layers[CN_NLAYERS - 1];
        CLEAR_NET_ASSERT(player.type == Convolutional ||
                         player.type == Pooling);
        if (player.type == Convolutional) {
            input_nrows = player.data.conv.output_nrows;
            input_ncols = player.data.conv.output_ncols;
            nimput = player.data.conv.nfilters;
        } else {
            input_nrows = player.data.pooling.output_nrows;
            input_ncols = player.data.pooling.output_ncols;
            nimput = player.data.pooling.noutput;
        }
    } else {
        if (net->input_ids.type == Mat) {
            input_nrows = net->input_ids.data.mat.nrows;
            input_ncols = net->input_ids.data.mat.ncols;
            nimput = net->input_ids.nimput;
        }
    }
    size_t output_nrows;
    size_t output_ncols;
    switch (padding) {
    case Same:
        output_nrows = input_nrows;
        output_ncols = input_ncols;
        break;
    case Full:
        output_nrows = input_nrows + kernel_nrows - 1;
        output_ncols = input_ncols + kernel_ncols - 1;
        break;
    case Valid:
        output_nrows = input_nrows - kernel_nrows + 1;
        output_ncols = input_ncols - kernel_ncols + 1;
        break;
    }

    CLEAR_NET_ASSERT(output_nrows > 0 && output_ncols > 0);

    ConvolutionalLayer conv_layer;
    conv_layer.nimput = nimput;
    conv_layer.nfilters = noutput;
    conv_layer.padding = padding;
    conv_layer.act = act;
    conv_layer.input_nrows = input_nrows;
    conv_layer.input_ncols = input_ncols;
    conv_layer.output_nrows = output_nrows;
    conv_layer.output_ncols = output_ncols;
    conv_layer.k_nrows = kernel_nrows;
    conv_layer.k_ncols = kernel_ncols;

    conv_layer.filters =
        CLEAR_NET_ALLOC(conv_layer.nfilters * sizeof(*conv_layer.filters));
    conv_layer.outputs =
        CLEAR_NET_ALLOC(conv_layer.nfilters * sizeof(*conv_layer.outputs));
    conv_layer.output_ids =
        CLEAR_NET_ALLOC(conv_layer.nfilters * sizeof(*conv_layer.output_ids));

    size_t offset = CN_NPARAMS + 1;

    for (size_t i = 0; i < conv_layer.nfilters; ++i) {
        Filter filter;
        filter.kernels =
            CLEAR_NET_ALLOC(conv_layer.nimput * sizeof(*filter.kernels));
        for (size_t j = 0; j < conv_layer.nimput; ++j) {
            filter.kernels[j] =
                cn_alloc_matrix(conv_layer.k_nrows, conv_layer.k_ncols);
            if (CN_WITH_MOMENTUM) {
                _cn_alloc_matrix_grad_stores(&filter.kernels[j]);
            }
            filter.kernels[j].gs_id = offset;
            offset += conv_layer.k_nrows * conv_layer.k_ncols;
        }
        filter.biases =
            cn_alloc_matrix(conv_layer.output_nrows, conv_layer.output_ncols);
        if (CN_WITH_MOMENTUM) {
            _cn_alloc_matrix_grad_stores(&filter.biases);
        }
        filter.biases.gs_id = offset;
        offset += conv_layer.output_nrows * conv_layer.output_ncols;
        conv_layer.filters[i] = filter;
        conv_layer.outputs[i] =
            cn_alloc_matrix(conv_layer.output_nrows, conv_layer.output_ncols);
        conv_layer.output_ids[i] =
            _cn_alloc_dmatrix(conv_layer.output_nrows, conv_layer.output_ncols);
        _cn_fill_params(conv_layer.outputs[i].elements,
                        conv_layer.output_nrows * conv_layer.output_ncols, 0);
    }
    CN_NPARAMS = offset - 1;
    Layer layer = _cn_init_layer(Conv);
    layer.data.conv = conv_layer;

    _cn_add_net_layer(net, layer, CN_NPARAMS);

    net->output_type = Mat;
}

void _cn_dealloc_conv_layer(ConvolutionalLayer *layer) {
    for (size_t i = 0; i < layer->nfilters; ++i) {
        Filter *cfilter = &layer->filters[i];
        for (size_t j = 0; j < layer->nimput; ++j) {
            cn_dealloc_matrix(&cfilter->kernels[j]);
        }
        CLEAR_NET_DEALLOC(cfilter->kernels);
        cn_dealloc_matrix(&cfilter->biases);
        cn_dealloc_matrix(&layer->outputs[i]);
    }

    cn_dealloc_matrix(layer->outputs);
    layer->filters = NULL;
    layer->outputs = NULL;
    layer->nfilters = 0;
    layer->padding = 0;
    layer->act = 0;
    layer->nimput = 0;
    layer->input_nrows = 0;
    layer->input_ncols = 0;
    layer->output_nrows = 0;
    layer->output_ncols = 0;
    layer->k_nrows = 0;
    layer->k_ncols = 0;
}

void _cn_randomize_conv_layer(ConvolutionalLayer *layer, scalar lower,
                              scalar upper) {
    for (size_t i = 0; i < layer->nfilters; ++i) {
        for (size_t j = 0; j < layer->nimput; ++j) {
            _cn_randomize_matrix(&layer->filters[i].kernels[j], lower, upper);
        }
        _cn_randomize_matrix(&layer->filters[i].biases, lower, upper);
    }
}

void _cn_copy_conv_params(GradientStore *gs, ConvolutionalLayer *layer) {
    for (size_t i = 0; i < layer->nfilters; ++i) {
        for (size_t j = 0; j < layer->nimput; ++j) {
            _cn_copy_matrix_params(gs, layer->filters[i].kernels[j]);
        }
        _cn_copy_matrix_params(gs, layer->filters[i].biases);
    }
}

scalar cn_correlate(Matrix kern, Matrix input, long top_left_row,
                    long top_left_col) {
    scalar res = 0;
    long lrows = (long)kern.nrows;
    long lcols = (long)kern.ncols;
    for (long i = 0; i < lrows; ++i) {
        for (long j = 0; j < lcols; ++j) {
            long r = top_left_row + i;
            long c = top_left_col + j;
            if (r >= 0 && c >= 0 && r < (long)input.nrows &&
                c < (long)input.ncols) {
                res += MAT_AT(input, r, c) * MAT_AT(kern, i, j);
            }
        }
    }

    return res;
}

Matrix *cn_forward_conv(ConvolutionalLayer *layer, Matrix *input) {
    for (size_t i = 0; i < layer->nfilters; ++i) {
        for (long j = 0; j < (long)layer->output_nrows; ++j) {
            for (long k = 0; k < (long)layer->output_ncols; ++k) {
                MAT_AT(layer->outputs[i], j, k) = 0;
            }
        }
    }

    size_t row_padding;
    size_t col_padding;
    _cn_set_padding(layer->padding, layer->k_nrows, layer->k_ncols,
                    &row_padding, &col_padding);

    for (size_t i = 0; i < layer->nimput; ++i) {
        for (size_t j = 0; j < layer->nfilters; ++j) {
            for (long k = 0; k < (long)layer->output_nrows; ++k) {
                for (long l = 0; l < (long)layer->output_ncols; ++l) {
                    long top_left_row = k - row_padding;
                    long top_left_col = l - col_padding;

                    scalar res =
                        cn_correlate(layer->filters[j].kernels[i], input[i],
                                     top_left_row, top_left_col);
                    MAT_AT(layer->outputs[j], k, l) += res;
                }
            }
        }
    }

    for (size_t i = 0; i < layer->nfilters; ++i) {
        for (size_t j = 0; j < layer->outputs[i].nrows; ++j) {
            for (size_t k = 0; k < layer->outputs[i].ncols; ++k) {
                MAT_AT(layer->outputs[i], j, k) +=
                    MAT_AT(layer->filters[i].biases, j, k);
                MAT_AT(layer->outputs[i], j, k) =
                    cn_activate(MAT_AT(layer->outputs[i], j, k), layer->act);
            }
        }
    }

    return layer->outputs;
}

size_t _cn_correlate(GradientStore *gs, Matrix kern, DMatrix input,
                     long top_left_row, long top_left_col) {
    size_t res = cn_init_leaf_var(gs, 0);
    long lrows = (long)kern.nrows;
    long lcols = (long)kern.ncols;
    for (long i = 0; i < lrows; ++i) {
        for (long j = 0; j < lcols; ++j) {
            long r = top_left_row + i;
            long c = top_left_col + j;
            if (r >= 0 && c >= 0 && r < (long)input.nrows &&
                c < (long)input.ncols) {
                size_t val =
                    cn_multiply(gs, MAT_AT(input, r, c), MAT_ID(kern, i, j));
                res = cn_add(gs, res, val);
            }
        }
    }

    return res;
}

DMatrix *_cn_forward_conv(ConvolutionalLayer *layer, GradientStore *gs,
                          DMatrix *input) {
    for (size_t i = 0; i < layer->nfilters; ++i) {
        for (size_t j = 0; j < layer->output_nrows; ++j) {
            for (size_t k = 0; k < layer->output_ncols; ++k) {
                MAT_AT(layer->output_ids[i], j, k) = cn_init_leaf_var(gs, 0);
            }
        }
    }

    size_t row_padding;
    size_t col_padding;

    _cn_set_padding(layer->padding, layer->k_nrows, layer->k_ncols,
                    &row_padding, &col_padding);

    for (size_t i = 0; i < layer->nimput; ++i) {
        for (size_t j = 0; j < layer->nfilters; ++j) {
            for (long k = 0; k < (long)layer->output_nrows; ++k) {
                for (long l = 0; l < (long)layer->output_ncols; ++l) {
                    long top_left_row = k - row_padding;
                    long top_left_col = l - col_padding;
                    size_t res =
                        _cn_correlate(gs, layer->filters[j].kernels[i],
                                      input[i], top_left_row, top_left_col);
                    MAT_AT(layer->output_ids[j], k, l) =
                        cn_add(gs, MAT_AT(layer->output_ids[j], k, l), res);
                }
            }
        }
    }

    for (size_t i = 0; i < layer->nfilters; ++i) {
        for (size_t j = 0; j < layer->output_nrows; ++j) {
            for (size_t k = 0; k < layer->output_ncols; ++k) {
                MAT_AT(layer->output_ids[i], j, k) =
                    cn_add(gs, MAT_ID(layer->filters[i].biases, j, k),
                           MAT_AT(layer->output_ids[i], j, k));
                MAT_AT(layer->output_ids[i], j, k) = _cn_activate(
                    gs, MAT_AT(layer->output_ids[i], j, k), layer->act);
            }
        }
    }

    return layer->output_ids;
}

void _cn_apply_conv_grads(GradientStore *gs, ConvolutionalLayer *layer) {
    for (size_t i = 0; i < layer->nfilters; ++i) {
        for (size_t j = 0; j < layer->nimput; ++j) {
            _cn_apply_matrix_grads(gs, &layer->filters[i].kernels[j]);
        }
        _cn_apply_matrix_grads(gs, &layer->filters[i].biases);
    }
}

void _cn_save_conv_layer_to_file(FILE *fp, ConvolutionalLayer *layer) {
    fwrite(&layer->act, sizeof(layer->act), 1, fp);
    fwrite(&layer->padding, sizeof(layer->padding), 1, fp);
    fwrite(&layer->k_nrows, sizeof(layer->k_nrows), 1, fp);
    fwrite(&layer->k_ncols, sizeof(layer->k_ncols), 1, fp);
    fwrite(&layer->nfilters, sizeof(layer->nfilters), 1, fp);
    for (size_t i = 0; i < layer->nfilters; ++i) {
        for (size_t j = 0; j < layer->nimput; ++j) {
            fwrite(layer->filters[i].kernels[j].elements,
                   sizeof(*layer->filters[i].kernels[j].elements),
                   layer->k_nrows * layer->k_ncols, fp);
        }
        fwrite(layer->filters[i].biases.elements,
               sizeof(*layer->filters[i].biases.elements),
               layer->output_nrows * layer->output_ncols, fp);
    }
}

void _cn_alloc_conv_layer_from_file(FILE *fp, Net *net) {
    Activation act;
    fread(&act, sizeof(act), 1, fp);
    Padding padding;
    fread(&padding, sizeof(padding), 1, fp);
    size_t k_nrows;
    fread(&k_nrows, sizeof(k_nrows), 1, fp);
    size_t k_ncols;
    fread(&k_ncols, sizeof(k_ncols), 1, fp);
    size_t nfilters;
    fread(&nfilters, sizeof(nfilters), 1, fp);

    cn_alloc_conv_layer(net, padding, act, nfilters, k_nrows, k_ncols);
    ConvolutionalLayer *layer = &net->layers[CN_NLAYERS - 1].data.conv;
    for (size_t i = 0; i < nfilters; ++i) {
        for (size_t j = 0; j < layer->nimput; ++j) {
            Matrix kern = layer->filters[i].kernels[j];
            fread(kern.elements, sizeof(*kern.elements), k_nrows * k_ncols, fp);
        }
        Matrix bias = layer->filters[i].biases;
        fread(bias.elements, sizeof(*bias.elements),
              layer->output_nrows * layer->output_ncols, fp);
    }
}

void _cn_print_conv_layer(ConvolutionalLayer *layer, size_t index) {
    printf("Layer #%zu: Convolutional\n", index);
    for (size_t i = 0; i < layer->nfilters; ++i) {
        printf("Filter #%zu\n", i);
        for (size_t j = 0; j < layer->nimput; ++j) {
            printf("kernel #%zu", j);
            cn_print_matrix(layer->filters[i].kernels[j], "");
        }
        cn_print_matrix(layer->filters[i].biases, "  filter bias");
    }
}

/* Implement: Pooling Layer */
void cn_alloc_pooling_layer(Net *net, PoolingStrategy strat,
                            size_t kernel_nrows, size_t kernel_ncols) {
    CLEAR_NET_ASSERT(net->layers[CN_NLAYERS - 1].type == Conv);
    PoolingLayer pooler;
    pooler.strat = strat;
    pooler.k_nrows = kernel_nrows;
    pooler.k_ncols = kernel_ncols;
    pooler.output_nrows =
        net->layers[CN_NLAYERS - 1].data.conv.output_nrows / kernel_nrows;
    pooler.output_ncols =
        net->layers[CN_NLAYERS - 1].data.conv.output_ncols / kernel_ncols;
    size_t nimput = net->layers[CN_NLAYERS - 1].data.conv.nfilters;
    pooler.outputs = CLEAR_NET_ALLOC(nimput * sizeof(Matrix));
    pooler.output_ids = CLEAR_NET_ALLOC(nimput * sizeof(DMatrix));
    for (size_t i = 0; i < nimput; ++i) {
        pooler.outputs[i] =
            cn_alloc_matrix(pooler.output_nrows, pooler.output_ncols);
        pooler.output_ids[i] =
            _cn_alloc_dmatrix(pooler.output_nrows, pooler.output_ncols);
    }
    pooler.noutput = nimput;

    Layer layer = _cn_init_layer(Pooling);
    layer.data.pooling = pooler;

    _cn_add_net_layer(net, layer, CN_NPARAMS);
    net->output_type = Mat;
}

void _cn_dealloc_pooling_layer(PoolingLayer *layer) {
    layer->strat = 0;
    layer->k_nrows = 0;
    layer->k_ncols = 0;
    layer->output_nrows = 0;
    layer->output_ncols = 0;
    for (size_t i = 0; i < layer->noutput; ++i) {
        cn_dealloc_matrix(&layer->outputs[i]);
        _cn_dealloc_dmatrix(&layer->output_ids[i]);
    }
    CLEAR_NET_DEALLOC(layer->outputs);
    CLEAR_NET_DEALLOC(layer->output_ids);
    layer->noutput = 0;
}

Matrix *cn_pool_layer(PoolingLayer *pooler, Matrix *input) {
    size_t nelements = pooler->k_nrows * pooler->k_ncols;
    for (size_t i = 0; i < pooler->noutput; ++i) {
        for (size_t j = 0; j < input[i].nrows; j += pooler->k_nrows) {
            for (size_t k = 0; k < input[i].ncols; k += pooler->k_ncols) {
                scalar max_store = -1 * CN_MAX_PARAM;
                scalar avg_store = 0;
                scalar cur;
                for (size_t l = 0; l < pooler->k_nrows; ++l) {
                    for (size_t m = 0; m < pooler->k_ncols; ++m) {
                        cur = MAT_AT(input[i], j + l, k + m);
                        switch (pooler->strat) {
                        case Max:
                            if (cur > max_store) {
                                max_store = cur;
                            }
                            break;
                        case Average:
                            avg_store += cur;
                            break;
                        }
                    }
                }
                switch (pooler->strat) {
                case Max:
                    MAT_AT(pooler->outputs[i], j / pooler->k_nrows,
                           k / pooler->k_ncols) = max_store;
                    break;
                case Average:
                    MAT_AT(pooler->outputs[i], j / pooler->k_nrows,
                           k / pooler->k_ncols) = avg_store / nelements;
                    break;
                }
            }
        }
    }

    return pooler->outputs;
}

DMatrix *_cn_pool_layer(GradientStore *gs, PoolingLayer *pooler,
                        DMatrix *input) {
    for (size_t i = 0; i < pooler->noutput; ++i) {
        for (size_t j = 0; j < input[i].nrows; j += pooler->k_nrows) {
            for (size_t k = 0; k < input[i].ncols; k += pooler->k_ncols) {
                scalar max_store = -1 * CN_MAX_PARAM;
                size_t max_id = 0;
                scalar avg_id = cn_init_leaf_var(gs, 0);
                size_t cur;
                size_t nelements = pooler->k_nrows * pooler->k_ncols;
                for (size_t l = 0; l < pooler->k_nrows; ++l) {
                    for (size_t m = 0; m < pooler->k_ncols; ++m) {
                        cur = MAT_AT(input[i], j + l, k + m);
                        switch (pooler->strat) {
                        case (Max):
                            if (GET_NODE(cur).num > max_store) {
                                max_store = GET_NODE(cur).num;
                                max_id = cur;
                            }
                            break;
                        case (Average):
                            avg_id = cn_add(gs, avg_id, cur);
                            break;
                        }
                    }
                }
                switch (pooler->strat) {
                case (Max):
                    MAT_AT(pooler->output_ids[i], j / pooler->k_nrows,
                           k / pooler->k_ncols) = max_id;
                    break;
                case (Average): {
                    size_t coef = cn_init_leaf_var(gs, 1 / (scalar)nelements);
                    MAT_AT(pooler->output_ids[i], j / pooler->k_nrows,
                           k / pooler->k_ncols) = cn_multiply(gs, avg_id, coef);
                    break;
                }
                }
            }
        }
    }

    return pooler->output_ids;
}

void _cn_save_pooling_layer_to_file(FILE *fp, PoolingLayer *layer) {
    fwrite(&layer->strat, sizeof(layer->strat), 1, fp);
    fwrite(&layer->k_nrows, sizeof(layer->k_nrows), 1, fp);
    fwrite(&layer->k_ncols, sizeof(layer->k_ncols), 1, fp);
}

void _cn_alloc_pooling_layer_from_file(FILE *fp, Net *net) {
    PoolingStrategy strat;
    fread(&strat, sizeof(strat), 1, fp);
    size_t k_nrows;
    fread(&k_nrows, sizeof(k_nrows), 1, fp);
    size_t k_ncols;
    fread(&k_ncols, sizeof(k_ncols), 1, fp);
    cn_alloc_pooling_layer(net, strat, k_nrows, k_ncols);
}

void _cn_print_pooling_layer(PoolingLayer layer, size_t layer_id) {
    printf("Layer #%zu: ", layer_id);
    switch (layer.strat) {
    case (Max):
        printf("Max ");
        break;
    case (Average):
        printf("Average ");
        break;
    }
    printf("Pooling Layer\n");
}

void cn_alloc_global_pooling_layer(Net *net, PoolingStrategy strat) {
    CLEAR_NET_ASSERT(CN_NLAYERS > 0);
    CLEAR_NET_ASSERT(net->layers[CN_NLAYERS - 1].type == Conv ||
                     net->layers[CN_NLAYERS - 1].type == Pooling);

    GlobalPoolingLayer gpooler = (GlobalPoolingLayer){
        .strat = strat,
        .output =
            cn_alloc_vector(net->layers[CN_NLAYERS - 1].data.conv.nfilters),
        .output_ids =
            _cn_alloc_dvector(net->layers[CN_NLAYERS - 1].data.conv.nfilters),
    };
    Layer layer = _cn_init_layer(GlobalPooling);
    layer.data.global_pooling = gpooler;

    _cn_add_net_layer(net, layer, CN_NPARAMS);
    net->output_type = Vec;
}

void _cn_dealloc_global_pooling_layer(GlobalPoolingLayer *layer) {
    cn_dealloc_vector(&layer->output);
    layer->strat = 0;
}

Vector cn_global_pool_layer(GlobalPoolingLayer *pooler, Matrix *input) {
    for (size_t i = 0; i < pooler->output.nelem; ++i) {
        scalar max_store = -1 * CN_MAX_PARAM;
        scalar avg_res = 0;
        scalar cur;
        size_t nelements = input[i].nrows * input[i].ncols;
        for (size_t j = 0; j < input[i].nrows; ++j) {
            for (size_t k = 0; k < input[i].ncols; ++k) {
                cur = MAT_AT(input[i], j, k);
                switch (pooler->strat) {
                case (Max):
                    if (cur > max_store) {
                        max_store = cur;
                    }
                    break;
                case (Average):
                    avg_res += cur;
                    break;
                }
            }
        }
        switch (pooler->strat) {
        case (Max):
            VEC_AT(pooler->output, i) = max_store;
            break;
        case (Average):
            VEC_AT(pooler->output, i) = avg_res / nelements;
            break;
        }
    }
    return pooler->output;
}

DVector _cn_global_pool_layer(GradientStore *gs, GlobalPoolingLayer *pooler,
                              DMatrix *input) {
    for (size_t i = 0; i < pooler->output.nelem; ++i) {
        scalar max_store = -1 * CN_MAX_PARAM;
        size_t max_id;
        size_t avg_id = cn_init_leaf_var(gs, 0);
        size_t nelements = input[i].nrows * input[i].ncols;
        for (size_t j = 0; j < input[i].nrows; ++j) {
            for (size_t k = 0; k < input[i].ncols; ++k) {
                size_t cur = MAT_AT(input[i], j, k);
                switch (pooler->strat) {
                case (Max):
                    if (GET_NODE(cur).num > max_store) {
                        max_store = GET_NODE(cur).num;
                        max_id = cur;
                    }
                    break;
                case (Average):
                    avg_id = cn_add(gs, avg_id, cur);
                    break;
                }
            }
        }
        switch (pooler->strat) {
        case (Max):
            VEC_AT(pooler->output_ids, i) = max_id;
            break;
        case (Average): {
            size_t coef = cn_init_leaf_var(gs, 1 / (scalar)nelements);
            VEC_AT(pooler->output_ids, i) = cn_multiply(gs, avg_id, coef);
            break;
        }
        }
    }

    return pooler->output_ids;
}

void _cn_save_global_pooling_layer_to_file(FILE *fp,
                                           GlobalPoolingLayer *layer) {
    fwrite(&layer->strat, sizeof(layer->strat), 1, fp);
}
void _cn_alloc_global_pooling_layer_from_file(FILE *fp, Net *net) {
    PoolingStrategy strat;
    fread(&strat, sizeof(strat), 1, fp);
    cn_alloc_global_pooling_layer(net, strat);
}

void _cn_print_global_pooling_layer(GlobalPoolingLayer gpooler,
                                    size_t layer_id) {
    printf("Layer #%zu: ", layer_id);
    switch (gpooler.strat) {
    case (Max):
        printf("Max ");
        break;
    case (Average):
        printf("Average ");
        break;
    }
    printf("Global Pooling Layer\n");
}

/* Implement: Layer */
Layer _cn_init_layer(LayerType type) {
    return (Layer){
        .type = type,
    };
}

/* Implement: Net */
Net cn_alloc_conv_net(size_t input_nrows, size_t input_ncols,
                      size_t nchannels) {
    CLEAR_NET_ASSERT(input_nrows != 0 && input_ncols != 0 && nchannels != 0);
    CN_NET_TYPE = Convolutional;
    DLAData input_ids;
    input_ids.nimput = nchannels;
    if (nchannels == 1) {
        input_ids.type = Mat;
        input_ids.data.mat = _cn_alloc_dmatrix(input_nrows, input_ncols);
    } else {
        input_ids.type = MatList;
        input_ids.data.mat_list = CLEAR_NET_ALLOC(nchannels * sizeof(DMatrix));
        for (size_t i = 0; i < nchannels; ++i) {
            input_ids.data.mat_list[i] =
                _cn_alloc_dmatrix(input_nrows, input_ncols);
        }
    }
    return (Net){
        .input_ids = input_ids,
        .computation_graph = cn_alloc_gradient_store(0),
        .layers = NULL,
    };
}

Net cn_alloc_vani_net(size_t input_nelem) {
    CLEAR_NET_ASSERT(input_nelem != 0);
    CN_NET_TYPE = Vanilla;
    DLAData input_ids;
    input_ids.type = Vec;
    input_ids.nimput = 1;
    input_ids.data.vec = _cn_alloc_dvector(input_nelem);
    return (Net){
        .input_ids = input_ids,
        .computation_graph = cn_alloc_gradient_store(0),
        .layers = NULL,
    };
}

void cn_dealloc_net(Net *net) {
    for (size_t i = 0; i < CN_NLAYERS - 1; ++i) {
        Layer layer = net->layers[i];
        switch (layer.type) {
        case (Dense):
            _cn_dealloc_dense_layer(&layer.data.dense);
            break;
        case (Convolutional):
            _cn_dealloc_conv_layer(&layer.data.conv);
            break;
        case (Pooling):
            break;
        case (GlobalPooling):
            break;
        }
    }
    cn_dealloc_gradient_store(&net->computation_graph);
    CLEAR_NET_DEALLOC(net->layers);
    _cn_dealloc_dladata(&net->input_ids);
    cn_default_hparams();
}

void _cn_add_net_layer(Net *net, Layer layer, size_t nparams) {
    net->layers =
        CLEAR_NET_REALLOC(net->layers, (CN_NLAYERS + 1) * sizeof(*net->layers));
    net->layers[CN_NLAYERS] = layer;
    CN_NLAYERS++;
    if (nparams != net->computation_graph.max_length) {
        cn_realloc_gradient_store(&net->computation_graph, CN_NPARAMS);
    }
}

void _cn_copy_net_params(Net *net, GradientStore *gs) {
    for (size_t i = 0; i < CN_NLAYERS; ++i) {
        Layer clayer = net->layers[i];
        if (clayer.type == Dense) {
            _cn_copy_dense_params(gs, &clayer.data.dense);
        } else if (clayer.type == Conv) {
            _cn_copy_conv_params(gs, &clayer.data.conv);
        }
    }
}

void cn_randomize_net(Net *net, scalar lower, scalar upper) {
    for (size_t i = 0; i < CN_NLAYERS; ++i) {
        Layer layer = net->layers[i];
        if (layer.type == Dense) {
            _cn_randomize_dense_layer(&layer.data.dense, lower, upper);
        } else if (layer.type == Conv) {
            _cn_randomize_conv_layer(&layer.data.conv, lower, upper);
        }
    }
}

void cn_shuffle_vani_input(Matrix *input, Matrix *target) {
    CLEAR_NET_ASSERT(input->nrows == target->nrows);
    scalar t;
    for (size_t i = 0; i < input->nrows; ++i) {
        size_t j = i + rand() % (input->nrows - i);
        if (i != j) {
            for (size_t k = 0; k < input->ncols; ++k) {
                t = MAT_AT(*input, i, k);
                MAT_AT(*input, i, k) = MAT_AT(*input, j, k);
                MAT_AT(*input, j, k) = t;
            }
            for (size_t k = 0; k < target->ncols; ++k) {
                t = MAT_AT(*target, i, k);
                MAT_AT(*target, i, k) = MAT_AT(*target, j, k);
                MAT_AT(*target, j, k) = t;
            }
        }
    }
}

void cn_get_batch_vani(Matrix *batch_in, Matrix *batch_tar, Matrix all_input,
                       Matrix all_target, size_t batch_num, size_t batch_size) {
    *batch_in = cn_form_matrix(batch_size, all_input.ncols, all_input.stride,
                               &MAT_AT(all_input, batch_num * batch_size, 0));
    *batch_tar = cn_form_matrix(batch_size, all_target.ncols, all_target.stride,
                                &MAT_AT(all_target, batch_num * batch_size, 0));
}

void cn_shuffle_conv_input(Matrix ***input, LAData **targets, size_t len) {
    for (size_t i = 0; i < len; i++) {
        size_t j = i + rand() % (len - i);
        Matrix *imp = (*input)[i];
        (*input)[i] = (*input)[j];
        (*input)[j] = imp;
        LAData tmp = (*targets)[i];
        (*targets)[i] = (*targets)[j];
        (*targets)[j] = tmp;
    }
}

void cn_get_batch_conv(Matrix **batch_in, LAData *batch_tar, Matrix **all_input,
                       LAData *all_target, size_t batch_num,
                       size_t batch_size) {
    size_t shift = batch_num * batch_size;
    for (size_t i = 0; i < batch_size; ++i) {
        batch_in[i] = all_input[i + shift];
        batch_tar[i] = all_target[i + shift];
    }
}

void cn_save_net_to_file(Net net, char *file_name) {
    FILE *fp = fopen(file_name, "wb");
    fwrite(&CN_NET_TYPE, sizeof(CN_NET_TYPE), 1, fp);
    fwrite(&net.input_ids.nimput, sizeof(net.input_ids.nimput), 1, fp);

    switch (CN_NET_TYPE) {
    case (Vanilla):
        fwrite(&net.input_ids.data.vec.nelem,
               sizeof(net.input_ids.data.vec.nelem), 1, fp);
        break;
    case (Conv):
        if (net.input_ids.type == Mat) {
            fwrite(&net.input_ids.data.mat.nrows,
                   sizeof(net.input_ids.data.mat.nrows), 1, fp);
            fwrite(&net.input_ids.data.mat.ncols,
                   sizeof(net.input_ids.data.mat.ncols), 1, fp);
        } else {
            fwrite(&net.input_ids.data.mat_list->nrows,
                   sizeof(net.input_ids.data.mat_list->nrows), 1, fp);
            fwrite(&net.input_ids.data.mat_list->ncols,
                   sizeof(net.input_ids.data.mat_list->ncols), 1, fp);
        }
        break;
    }
    fwrite(&CN_RATE, sizeof(CN_RATE), 1, fp);
    fwrite(&CN_NLAYERS, sizeof(CN_NLAYERS), 1, fp);
    fwrite(&CN_NEG_SCALE, sizeof(CN_NEG_SCALE), 1, fp);
    fwrite(&CN_WITH_MOMENTUM, sizeof(CN_WITH_MOMENTUM), 1, fp);
    fwrite(&CN_MOMENTUM_BETA, sizeof(CN_MOMENTUM_BETA), 1, fp);
    for (size_t i = 0; i < CN_NLAYERS; ++i) {
        fwrite(&net.layers[i].type, sizeof(net.layers[i].type), 1, fp);
        switch (net.layers[i].type) {
        case (Dense):
            _cn_save_dense_layer_to_file(fp, &net.layers[i].data.dense);
            break;
        case (Convolutional):
            _cn_save_conv_layer_to_file(fp, &net.layers[i].data.conv);
            break;
        case (Pooling):
            _cn_save_pooling_layer_to_file(fp, &net.layers[i].data.pooling);
            break;
        case (GlobalPooling):
            _cn_save_global_pooling_layer_to_file(
                fp, &net.layers[i].data.global_pooling);
            break;
        }
    }
    fclose(fp);
}

Net cn_alloc_net_from_file(char *file_name) {
    FILE *fp = fopen(file_name, "rb");
    CLEAR_NET_ASSERT(fp != NULL);
    NetType type;
    fread(&type, sizeof(type), 1, fp);
    size_t nimput;
    fread(&nimput, sizeof(nimput), 1, fp);
    Net net;
    switch (type) {
    case (Vanilla): {
        size_t nelem;
        fread(&nelem, sizeof(nelem), 1, fp);
        net = cn_alloc_vani_net(nelem);
        break;
    }
    case (Conv): {
        size_t input_nrows;
        fread(&input_nrows, sizeof(input_nrows), 1, fp);
        size_t input_ncols;
        fread(&input_ncols, sizeof(input_ncols), 1, fp);
        net = cn_alloc_conv_net(input_nrows, input_ncols, nimput);
        break;
    }
    }
    fread(&CN_RATE, sizeof(CN_RATE), 1, fp);
    size_t nlayers;
    fread(&nlayers, sizeof(CN_NLAYERS), 1, fp);
    fread(&CN_NEG_SCALE, sizeof(CN_NEG_SCALE), 1, fp);
    fread(&CN_WITH_MOMENTUM, sizeof(CN_WITH_MOMENTUM), 1, fp);
    fread(&CN_MOMENTUM_BETA, sizeof(CN_MOMENTUM_BETA), 1, fp);
    LayerType ctype;
    for (size_t i = 0; i < nlayers; ++i) {
        fread(&ctype, sizeof(ctype), 1, fp);
        switch (ctype) {
        case (Dense):
            _cn_alloc_dense_layer_from_file(fp, &net);
            break;
        case (Convolutional):
            _cn_alloc_conv_layer_from_file(fp, &net);
            break;
        case (Pooling):
            _cn_alloc_pooling_layer_from_file(fp, &net);
            break;
        case (GlobalPooling):
            _cn_alloc_global_pooling_layer_from_file(fp, &net);
            break;
        }
    }
    CLEAR_NET_ASSERT(nlayers == CN_NLAYERS);
    fclose(fp);
    return net;
}

void cn_print_net(Net net, char *name) {
    printf("%s = [\n", name);
    for (size_t i = 0; i < CN_NLAYERS; ++i) {
        Layer layer = net.layers[i];
        switch (layer.type) {
        case (Dense):
            _cn_print_dense(layer.data.dense, i);
            break;
        case (Convolutional):
            _cn_print_conv_layer(&layer.data.conv, i);
            break;
        case (Pooling):
            break;
        case (GlobalPooling):
            _cn_print_global_pooling_layer(layer.data.global_pooling, i);
            break;
        }
    }
    printf("]\n");
}

/* Implement: Vanilla */
scalar cn_learn_vani(Net *net, Matrix input, Matrix target) {
    CLEAR_NET_ASSERT(input.nrows == target.nrows);
    size_t train_size = input.nrows;
    net->computation_graph.length = 1;
    GradientStore *gs = &net->computation_graph;

    _cn_copy_net_params(net, gs);
    for (size_t i = 0; i < input.ncols; ++i) {
        VEC_AT(net->input_ids.data.vec, i) = gs->length + i;
    }
    scalar total_loss = 0;
    Vector input_vec;
    Vector target_vec;
    for (size_t i = 0; i < train_size; ++i) {
        input_vec = cn_form_vector(input.ncols, &MAT_AT(input, i, 0));
        target_vec = cn_form_vector(target.ncols, &MAT_AT(target, i, 0));
        total_loss += _cn_find_grad_vani(net, gs, input_vec, target_vec);
        gs->length = CN_NPARAMS + 1;
    }

    for (size_t i = 0; i < CN_NLAYERS; ++i) {
        _cn_apply_dense_grads(gs, &net->layers[i].data.dense);
    }

    return total_loss / train_size;
}

scalar _cn_find_grad_vani(Net *net, GradientStore *gs, Vector input,
                          Vector target) {
    input.gs_id = gs->length;
    _cn_copy_vector_params(gs, input);
    DVector prediction = _cn_predict_vani(net, gs, net->input_ids.data.vec);

    target.gs_id = gs->length;
    _cn_copy_vector_params(gs, target);
    size_t raiser = cn_init_leaf_var(gs, 2);
    size_t loss = cn_init_leaf_var(gs, 0);
    for (size_t i = 0; i < target.nelem; ++i) {
        loss = cn_add(
            gs, loss,
            cn_raise(gs,
                     cn_subtract(gs, VEC_AT(prediction, i), VEC_ID(target, i)),
                     raiser));
    }
    cn_backward(gs, loss);

    return GET_NODE(loss).num;
}

Vector cn_predict_vani(Net *net, Vector input) {
    Vector out = input;

    for (size_t i = 0; i < CN_NLAYERS; ++i) {
        out = cn_forward_dense(&net->layers[i].data.dense, out);
    }

    return out;
}

DVector _cn_predict_vani(Net *net, GradientStore *gs, DVector prev_ids) {
    for (size_t i = 0; i < CN_NLAYERS; ++i) {
        prev_ids = _cn_forward_dense(&net->layers[i].data.dense, gs, prev_ids);
    }
    return prev_ids;
}

scalar cn_loss_vani(Net *net, Matrix input, Matrix target) {
    CLEAR_NET_ASSERT(input.nrows == target.nrows);
    size_t size = input.nrows;
    scalar loss = 0;
    for (size_t i = 0; i < size; ++i) {
        Vector in = cn_form_vector(input.ncols, &MAT_AT(input, i, 0));
        Vector tar = cn_form_vector(target.ncols, &MAT_AT(target, i, 0));
        Vector out = cn_predict_vani(net, in);
        for (size_t j = 0; j < out.nelem; ++j) {
            loss += powf(VEC_AT(out, j) - VEC_AT(tar, j), 2);
        }
    }
    return loss / size;
}

void cn_print_vani_results(Net net, Matrix input, Matrix target) {
    CLEAR_NET_ASSERT(input.nrows == target.nrows);
    size_t size = input.nrows;
    printf("Input | Net Output | Target\n");
    scalar loss = 0;
    for (size_t i = 0; i < size; ++i) {
        Vector in = cn_form_vector(input.ncols, &MAT_AT(input, i, 0));
        Vector tar = cn_form_vector(target.ncols, &MAT_AT(target, i, 0));
        Vector out = cn_predict_vani(&net, in);
        for (size_t j = 0; j < out.nelem; ++j) {
            loss += powf(VEC_AT(out, j) - VEC_AT(tar, j), 2);
        }
        cn_print_vector_inline(in);
        printf("| ");
        cn_print_vector_inline(out);
        printf("| ");
        cn_print_vector_inline(tar);
        printf("| ");
        printf("\n");
    }
    printf("Average Loss: %f\n", loss / size);
}

void cn_print_target_output_pairs_vani(Net net, Matrix input, Matrix target) {
    CLEAR_NET_ASSERT(input.nrows == target.nrows);
    Vector in;
    Vector tar;
    Vector out;
    for (size_t i = 0; i < input.nrows; ++i) {
        in = cn_form_vector(input.ncols, &MAT_AT(input, i, 0));
        tar = cn_form_vector(target.ncols, &MAT_AT(target, i, 0));
        out = cn_predict_vani(&net, in);
        printf("------------\n");
        printf("target: ");
        for (size_t j = 0; j < tar.nelem; ++j) {
            printf("%f ", VEC_AT(tar, j));
        }
        printf("\n");
        printf("output: ");
        for (size_t j = 0; j < out.nelem; ++j) {
            printf("%f ", VEC_AT(out, j));
        }
        printf("\n");
    }
}

/* Implement: Convolutional Net */
scalar cn_learn_conv(Net *net, Matrix **inputs, LAData *targets,
                     size_t nimput) {
    CLEAR_NET_ASSERT((*targets).type == net->output_type);
    scalar total_loss = 0;
    net->computation_graph.length = 1;
    GradientStore *gs = &net->computation_graph;

    _cn_copy_net_params(net, gs);
    size_t offset = gs->length;
    if (net->layers[0].data.conv.nimput == 1) {
        for (size_t i = 0; i < net->input_ids.data.mat.nrows; ++i) {
            for (size_t j = 0; j < net->input_ids.data.mat.ncols; ++j) {
                MAT_AT(net->input_ids.data.mat, i, j) = offset++;
            }
        }
    } else {
        for (size_t i = 0; i < net->layers[0].data.conv.nimput; ++i) {
            for (size_t j = 0; j < net->input_ids.data.mat_list->nrows; ++j) {
                for (size_t k = 0; k < net->input_ids.data.mat_list->ncols;
                     ++k) {
                    MAT_AT(net->input_ids.data.mat_list[i], j, k) = offset++;
                }
            }
        }
    }

    for (size_t i = 0; i < nimput; ++i) {
        total_loss += _cn_find_grad_conv(net, gs, inputs[i], targets[i]);
        gs->length = CN_NPARAMS + 1;
    }

    for (size_t i = 0; i < CN_NLAYERS; ++i) {
        if (net->layers[i].type == Dense) {
            _cn_apply_dense_grads(gs, &net->layers[i].data.dense);
        } else if (net->layers[i].type == Conv) {
            _cn_apply_conv_grads(gs, &net->layers[i].data.conv);
        }
    }

    return total_loss / nimput;
}

scalar _cn_find_grad_conv(Net *net, GradientStore *gs, Matrix *input,
                          LAData target) {
    for (size_t i = 0; i < net->layers[0].data.conv.nimput; ++i) {
        input[i].gs_id = gs->length;
        _cn_copy_matrix_params(gs, input[i]);
    }
    DLAData prediction;
    if (net->layers[0].data.conv.nimput == 1) {
        prediction = _cn_predict_conv(net, gs, &net->input_ids.data.mat);
    } else {
        prediction = _cn_predict_conv(net, gs, net->input_ids.data.mat_list);
    }
    size_t raiser = cn_init_leaf_var(gs, 2);
    switch (prediction.type) {
    case (Vec): {
        target.data.vec.gs_id = gs->length;
        _cn_copy_vector_params(gs, target.data.vec);
        size_t loss = cn_init_leaf_var(gs, 0);
        for (size_t i = 0; i < target.data.vec.nelem; ++i) {
            loss =
                cn_add(gs, loss,
                       cn_raise(gs,
                                cn_subtract(gs, VEC_AT(prediction.data.vec, i),
                                            VEC_ID(target.data.vec, i)),
                                raiser));
        }
        cn_backward(gs, loss);
        return GET_NODE(loss).num;
    }
    case (Mat):
        target.data.mat.gs_id = gs->length;
        _cn_copy_matrix_params(gs, target.data.mat);
        size_t loss = cn_init_leaf_var(gs, 0);
        for (size_t i = 0; i < target.data.mat.nrows; ++i) {
            for (size_t j = 0; j < target.data.mat.ncols; ++j) {
                loss = cn_add(
                    gs, loss,
                    cn_raise(gs,
                             cn_subtract(gs, MAT_AT(prediction.data.mat, i, j),
                                         MAT_AT(target.data.mat, i, j)),
                             raiser));
            }
        }
        cn_backward(gs, loss);
        return GET_NODE(loss).num;
    case (MatList): {
        Matrix *tar_list = target.data.mat_list;
        for (size_t i = 0; i < net->layers[CN_NLAYERS - 1].data.conv.nfilters;
             ++i) {
            for (size_t j = 0; j < tar_list[i].nrows; ++j) {
                for (size_t k = 0; k < tar_list[i].nrows; ++k) {
                    loss = cn_add(
                        gs, loss,
                        cn_raise(gs,
                                 cn_subtract(
                                     gs,
                                     MAT_AT(prediction.data.mat_list[i], j, k),
                                     MAT_AT(tar_list[i], j, k)),
                                 raiser));
                }
            }
        }
        cn_backward(gs, loss);
        return GET_NODE(loss).num;
    }
    }
}

LAData cn_predict_conv(Net *net, Matrix *minput) {
    CLEAR_NET_ASSERT(net->layers[0].type == Convolutional);
    Vector vinput;
    for (size_t i = 0; i < CN_NLAYERS; ++i) {
        Layer clayer = net->layers[i];
        switch (clayer.type) {
        case (Dense):
            vinput = cn_forward_dense(&clayer.data.dense, vinput);
            break;
        case (Convolutional):
            minput = cn_forward_conv(&clayer.data.conv, minput);
            break;
        case (Pooling):
            minput = cn_pool_layer(&clayer.data.pooling, minput);
            break;
        case (GlobalPooling):
            vinput = cn_global_pool_layer(&clayer.data.global_pooling, minput);
            break;
        }
    }

    LAData res;
    res.type = net->output_type;
    switch (res.type) {
    case (Vec):
        res.data.vec = vinput;
        break;
    case (Mat):
        res.data.mat = *minput;
        break;
    case (MatList):
        res.data.mat_list = minput;
        break;
    }

    return res;
}

DLAData _cn_predict_conv(Net *net, GradientStore *gs, DMatrix *minput) {
    CLEAR_NET_ASSERT(net->layers[0].type == Convolutional);
    DVector vinput;
    for (size_t i = 0; i < CN_NLAYERS; ++i) {
        Layer layer = net->layers[i];
        switch (layer.type) {
        case (Dense):
            vinput = _cn_forward_dense(&layer.data.dense, gs, vinput);
            break;
        case (Convolutional):
            minput = _cn_forward_conv(&layer.data.conv, gs, minput);
            break;
        case (Pooling):
            minput = _cn_pool_layer(gs, &layer.data.pooling, minput);
            break;
        case (GlobalPooling):
            vinput =
                _cn_global_pool_layer(gs, &layer.data.global_pooling, minput);
            break;
        }
    }

    DLAData res;
    res.type = net->output_type;
    switch (res.type) {
    case (Vec):
        res.data.vec = vinput;
        break;
    case (Mat):
        res.data.mat = *minput;
        break;
    case (MatList):
        res.data.mat_list = minput;
        break;
    }

    return res;
}

scalar cn_loss_conv(Net *net, Matrix **input, LAData *targets, size_t nimput) {
    scalar loss = 0;
    for (size_t i = 0; i < nimput; ++i) {
        LAData prediction = cn_predict_conv(net, input[i]);
        switch (prediction.type) {
        case (Vec): {
            for (size_t j = 0; j < prediction.data.vec.nelem; ++j) {
                loss += powf(VEC_AT(prediction.data.vec, j) -
                                 VEC_AT(targets[i].data.vec, j),
                             2);
            }
            break;
        }
        case (Mat): {
            for (size_t j = 0; j < prediction.data.mat.nrows; ++j) {
                for (size_t k = 0; k < prediction.data.mat.ncols; ++k) {
                    loss += powf(MAT_AT(prediction.data.mat, j, k) -
                                     MAT_AT(targets[i].data.mat, j, k),
                                 2);
                }
            }
            break;
        }
        case (MatList): {
            Matrix *list = prediction.data.mat_list;
            for (size_t j = 0;
                 j < net->layers[CN_NLAYERS - 1].data.conv.nfilters; ++j) {
                for (size_t k = 0; k < list[j].nrows; ++k) {
                    for (size_t l = 0; l < list[j].ncols; ++l) {
                        loss +=
                            powf(MAT_AT(list[j], k, l) -
                                     MAT_AT(targets[i].data.mat_list[j], k, l),
                                 2);
                    }
                }
            }
            break;
        }
        }
    }

    return loss / nimput;
}

void cn_print_conv_results(Net net, Matrix **inputs, LAData *targets,
                           size_t nimput) {
    size_t is_vec;
    switch ((*targets).type) {
    case Vec:
        is_vec = 1;
        break;
    case Mat:
        is_vec = 0;
        break;
    case MatList:
        CLEAR_NET_ASSERT(0 && "not supported");
    }
    for (size_t i = 0; i < nimput; ++i) {
        char *tar_name = "target";
        if (is_vec) {
            printf("%s: ", tar_name);
            cn_print_vector_inline(targets[i].data.vec);
        } else {
            cn_print_matrix(targets[i].data.mat, tar_name);
        }
        printf("\n");
        LAData pred = cn_predict_conv(&net, inputs[i]);
        char *out_name = "output";
        if (is_vec) {
            printf("%s: ", out_name);
            cn_print_vector_inline(pred.data.vec);
        } else {
            cn_print_matrix(pred.data.mat, out_name);
        }
        printf("\n");
        printf("--------------\n");
    }
}

#endif // CLEAR_NET_IMPLEMENTATION
/* Ending */

/* License
   Attribution 4.0 International

   =======================================================================

   Creative Commons Corporation ("Creative Commons") is not a law firm and
   does not provide legal services or legal advice. Distribution of
   Creative Commons public licenses does not create a lawyer-client or
   other relationship. Creative Commons makes its licenses and related
   information available on an "as-is" basis. Creative Commons gives no
   warranties regarding its licenses, any material licensed under their
   terms and conditions, or any related information. Creative Commons
   disclaims all liability for damages resulting from their use to the
   fullest extent possible.

   Using Creative Commons Public Licenses

   Creative Commons public licenses provide a standard set of terms and
   conditions that creators and other rights holders may use to share
   original works of authorship and other material subject to copyright
   and certain other rights specified in the public license below. The
   following considerations are for informational purposes only, are not
   exhaustive, and do not form part of our licenses.

   Considerations for licensors: Our public licenses are
   intended for use by those authorized to give the public
   permission to use material in ways otherwise restricted by
   copyright and certain other rights. Our licenses are
   irrevocable. Licensors should read and understand the terms
   and conditions of the license they choose before applying it.
   Licensors should also secure all rights necessary before
   applying our licenses so that the public can reuse the
   material as expected. Licensors should clearly mark any
   material not subject to the license. This includes other CC-
   licensed material, or material used under an exception or
   limitation to copyright. More considerations for licensors:
   wiki.creativecommons.org/Considerations_for_licensors

   Considerations for the public: By using one of our public
   licenses, a licensor grants the public permission to use the
   licensed material under specified terms and conditions. If
   the licensor's permission is not necessary for any reason--for
   example, because of any applicable exception or limitation to
   copyright--then that use is not regulated by the license. Our
   licenses grant only permissions under copyright and certain
   other rights that a licensor has authority to grant. Use of
   the licensed material may still be restricted for other
   reasons, including because others have copyright or other
   rights in the material. A licensor may make special requests,
   such as asking that all changes be marked or described.
   Although not required by our licenses, you are encouraged to
   respect those requests where reasonable. More considerations
   for the public:
   wiki.creativecommons.org/Considerations_for_licensees

   =======================================================================

   Creative Commons Attribution 4.0 International Public License

   By exercising the Licensed Rights (defined below), You accept and agree
   to be bound by the terms and conditions of this Creative Commons
   Attribution 4.0 International Public License ("Public License"). To the
   extent this Public License may be interpreted as a contract, You are
   granted the Licensed Rights in consideration of Your acceptance of
   these terms and conditions, and the Licensor grants You such rights in
   consideration of benefits the Licensor receives from making the
   Licensed Material available under these terms and conditions.


   Section 1 -- Definitions.

   a. Adapted Material means material subject to Copyright and Similar
   Rights that is derived from or based upon the Licensed Material
   and in which the Licensed Material is translated, altered,
   arranged, transformed, or otherwise modified in a manner requiring
   permission under the Copyright and Similar Rights held by the
   Licensor. For purposes of this Public License, where the Licensed
   Material is a musical work, performance, or sound recording,
   Adapted Material is always produced where the Licensed Material is
   synched in timed relation with a moving image.

   b. Adapter's License means the license You apply to Your Copyright
   and Similar Rights in Your contributions to Adapted Material in
   accordance with the terms and conditions of this Public License.

   c. Copyright and Similar Rights means copyright and/or similar rights
   closely related to copyright including, without limitation,
   performance, broadcast, sound recording, and Sui Generis Database
   Rights, without regard to how the rights are labeled or
   categorized. For purposes of this Public License, the rights
   specified in Section 2(b)(1)-(2) are not Copyright and Similar
   Rights.

   d. Effective Technological Measures means those measures that, in the
   absence of proper authority, may not be circumvented under laws
   fulfilling obligations under Article 11 of the WIPO Copyright
   Treaty adopted on December 20, 1996, and/or similar international
   agreements.

   e. Exceptions and Limitations means fair use, fair dealing, and/or
   any other exception or limitation to Copyright and Similar Rights
   that applies to Your use of the Licensed Material.

   f. Licensed Material means the artistic or literary work, database,
   or other material to which the Licensor applied this Public
   License.

   g. Licensed Rights means the rights granted to You subject to the
   terms and conditions of this Public License, which are limited to
   all Copyright and Similar Rights that apply to Your use of the
   Licensed Material and that the Licensor has authority to license.

   h. Licensor means the individual(s) or entity(ies) granting rights
   under this Public License.

   i. Share means to provide material to the public by any means or
   process that requires permission under the Licensed Rights, such
   as reproduction, public display, public performance, distribution,
   dissemination, communication, or importation, and to make material
   available to the public including in ways that members of the
   public may access the material from a place and at a time
   individually chosen by them.

   j. Sui Generis Database Rights means rights other than copyright
   resulting from Directive 96/9/EC of the European Parliament and of
   the Council of 11 March 1996 on the legal protection of databases,
   as amended and/or succeeded, as well as other essentially
   equivalent rights anywhere in the world.

   k. You means the individual or entity exercising the Licensed Rights
   under this Public License. Your has a corresponding meaning.


   Section 2 -- Scope.

   a. License grant.

   1. Subject to the terms and conditions of this Public License,
   the Licensor hereby grants You a worldwide, royalty-free,
   non-sublicensable, non-exclusive, irrevocable license to
   exercise the Licensed Rights in the Licensed Material to:

   a. reproduce and Share the Licensed Material, in whole or
   in part; and

   b. produce, reproduce, and Share Adapted Material.

   2. Exceptions and Limitations. For the avoidance of doubt, where
   Exceptions and Limitations apply to Your use, this Public
   License does not apply, and You do not need to comply with
   its terms and conditions.

   3. Term. The term of this Public License is specified in Section
   6(a).

   4. Media and formats; technical modifications allowed. The
   Licensor authorizes You to exercise the Licensed Rights in
   all media and formats whether now known or hereafter created,
   and to make technical modifications necessary to do so. The
   Licensor waives and/or agrees not to assert any right or
   authority to forbid You from making technical modifications
   necessary to exercise the Licensed Rights, including
   technical modifications necessary to circumvent Effective
   Technological Measures. For purposes of this Public License,
   simply making modifications authorized by this Section 2(a)
   (4) never produces Adapted Material.

   5. Downstream recipients.

   a. Offer from the Licensor -- Licensed Material. Every
   recipient of the Licensed Material automatically
   receives an offer from the Licensor to exercise the
   Licensed Rights under the terms and conditions of this
   Public License.

   b. No downstream restrictions. You may not offer or impose
   any additional or different terms or conditions on, or
   apply any Effective Technological Measures to, the
   Licensed Material if doing so restricts exercise of the
   Licensed Rights by any recipient of the Licensed
   Material.

   6. No endorsement. Nothing in this Public License constitutes or
   may be construed as permission to assert or imply that You
   are, or that Your use of the Licensed Material is, connected
   with, or sponsored, endorsed, or granted official status by,
   the Licensor or others designated to receive attribution as
   provided in Section 3(a)(1)(A)(i).

   b. Other rights.

   1. Moral rights, such as the right of integrity, are not
   licensed under this Public License, nor are publicity,
   privacy, and/or other similar personality rights; however, to
   the extent possible, the Licensor waives and/or agrees not to
   assert any such rights held by the Licensor to the limited
   extent necessary to allow You to exercise the Licensed
   Rights, but not otherwise.

   2. Patent and trademark rights are not licensed under this
   Public License.

   3. To the extent possible, the Licensor waives any right to
   collect royalties from You for the exercise of the Licensed
   Rights, whether directly or through a collecting society
   under any voluntary or waivable statutory or compulsory
   licensing scheme. In all other cases the Licensor expressly
   reserves any right to collect such royalties.


   Section 3 -- License Conditions.

   Your exercise of the Licensed Rights is expressly made subject to the
   following conditions.

   a. Attribution.

   1. If You Share the Licensed Material (including in modified
   form), You must:

   a. retain the following if it is supplied by the Licensor
   with the Licensed Material:

   i. identification of the creator(s) of the Licensed
   Material and any others designated to receive
   attribution, in any reasonable manner requested by
   the Licensor (including by pseudonym if
   designated);

   ii. a copyright notice;

   iii. a notice that refers to this Public License;

   iv. a notice that refers to the disclaimer of
   warranties;

   v. a URI or hyperlink to the Licensed Material to the
   extent reasonably practicable;

   b. indicate if You modified the Licensed Material and
   retain an indication of any previous modifications; and

   c. indicate the Licensed Material is licensed under this
   Public License, and include the text of, or the URI or
   hyperlink to, this Public License.

   2. You may satisfy the conditions in Section 3(a)(1) in any
   reasonable manner based on the medium, means, and context in
   which You Share the Licensed Material. For example, it may be
   reasonable to satisfy the conditions by providing a URI or
   hyperlink to a resource that includes the required
   information.

   3. If requested by the Licensor, You must remove any of the
   information required by Section 3(a)(1)(A) to the extent
   reasonably practicable.

   4. If You Share Adapted Material You produce, the Adapter's
   License You apply must not prevent recipients of the Adapted
   Material from complying with this Public License.


   Section 4 -- Sui Generis Database Rights.

   Where the Licensed Rights include Sui Generis Database Rights that
   apply to Your use of the Licensed Material:

   a. for the avoidance of doubt, Section 2(a)(1) grants You the right
   to extract, reuse, reproduce, and Share all or a substantial
   portion of the contents of the database;

   b. if You include all or a substantial portion of the database
   contents in a database in which You have Sui Generis Database
   Rights, then the database in which You have Sui Generis Database
   Rights (but not its individual contents) is Adapted Material; and

   c. You must comply with the conditions in Section 3(a) if You Share
   all or a substantial portion of the contents of the database.

   For the avoidance of doubt, this Section 4 supplements and does not
   replace Your obligations under this Public License where the Licensed
   Rights include other Copyright and Similar Rights.


   Section 5 -- Disclaimer of Warranties and Limitation of Liability.

   a. UNLESS OTHERWISE SEPARATELY UNDERTAKEN BY THE LICENSOR, TO THE
   EXTENT POSSIBLE, THE LICENSOR OFFERS THE LICENSED MATERIAL AS-IS
   AND AS-AVAILABLE, AND MAKES NO REPRESENTATIONS OR WARRANTIES OF
   ANY KIND CONCERNING THE LICENSED MATERIAL, WHETHER EXPRESS,
   IMPLIED, STATUTORY, OR OTHER. THIS INCLUDES, WITHOUT LIMITATION,
   WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR
   PURPOSE, NON-INFRINGEMENT, ABSENCE OF LATENT OR OTHER DEFECTS,
   ACCURACY, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR NOT
   KNOWN OR DISCOVERABLE. WHERE DISCLAIMERS OF WARRANTIES ARE NOT
   ALLOWED IN FULL OR IN PART, THIS DISCLAIMER MAY NOT APPLY TO YOU.

   b. TO THE EXTENT POSSIBLE, IN NO EVENT WILL THE LICENSOR BE LIABLE
   TO YOU ON ANY LEGAL THEORY (INCLUDING, WITHOUT LIMITATION,
   NEGLIGENCE) OR OTHERWISE FOR ANY DIRECT, SPECIAL, INDIRECT,
   INCIDENTAL, CONSEQUENTIAL, PUNITIVE, EXEMPLARY, OR OTHER LOSSES,
   COSTS, EXPENSES, OR DAMAGES ARISING OUT OF THIS PUBLIC LICENSE OR
   USE OF THE LICENSED MATERIAL, EVEN IF THE LICENSOR HAS BEEN
   ADVISED OF THE POSSIBILITY OF SUCH LOSSES, COSTS, EXPENSES, OR
   DAMAGES. WHERE A LIMITATION OF LIABILITY IS NOT ALLOWED IN FULL OR
   IN PART, THIS LIMITATION MAY NOT APPLY TO YOU.

   c. The disclaimer of warranties and limitation of liability provided
   above shall be interpreted in a manner that, to the extent
   possible, most closely approximates an absolute disclaimer and
   waiver of all liability.


   Section 6 -- Term and Termination.

   a. This Public License applies for the term of the Copyright and
   Similar Rights licensed here. However, if You fail to comply with
   this Public License, then Your rights under this Public License
   terminate automatically.

   b. Where Your right to use the Licensed Material has terminated under
   Section 6(a), it reinstates:

   1. automatically as of the date the violation is cured, provided
   it is cured within 30 days of Your discovery of the
   violation; or

   2. upon express reinstatement by the Licensor.

   For the avoidance of doubt, this Section 6(b) does not affect any
   right the Licensor may have to seek remedies for Your violations
   of this Public License.

   c. For the avoidance of doubt, the Licensor may also offer the
   Licensed Material under separate terms or conditions or stop
   distributing the Licensed Material at any time; however, doing so
   will not terminate this Public License.

   d. Sections 1, 5, 6, 7, and 8 survive termination of this Public
   License.


   Section 7 -- Other Terms and Conditions.

   a. The Licensor shall not be bound by any additional or different
   terms or conditions communicated by You unless expressly agreed.

   b. Any arrangements, understandings, or agreements regarding the
   Licensed Material not stated herein are separate from and
   independent of the terms and conditions of this Public License.


   Section 8 -- Interpretation.

   a. For the avoidance of doubt, this Public License does not, and
   shall not be interpreted to, reduce, limit, restrict, or impose
   conditions on any use of the Licensed Material that could lawfully
   be made without permission under this Public License.

   b. To the extent possible, if any provision of this Public License is
   deemed unenforceable, it shall be automatically reformed to the
   minimum extent necessary to make it enforceable. If the provision
   cannot be reformed, it shall be severed from this Public License
   without affecting the enforceability of the remaining terms and
   conditions.

   c. No term or condition of this Public License will be waived and no
   failure to comply consented to unless expressly agreed to by the
   Licensor.

   d. Nothing in this Public License constitutes or may be interpreted
   as a limitation upon, or waiver of, any privileges and immunities
   that apply to the Licensor or You, including from the legal
   processes of any jurisdiction or authority.


   =======================================================================

   Creative Commons is not a party to its public
   licenses. Notwithstanding, Creative Commons may elect to apply one of
   its public licenses to material it publishes and in those instances
   will be considered the “Licensor.” The text of the Creative Commons
   public licenses is dedicated to the public domain under the CC0 Public
   Domain Dedication. Except for the limited purpose of indicating that
   material is shared under a Creative Commons public license or as
   otherwise permitted by the Creative Commons policies published at
   creativecommons.org/policies, Creative Commons does not authorize the
   use of the trademark "Creative Commons" or any other trademark or logo
   of Creative Commons without its prior written consent including,
   without limitation, in connection with any unauthorized modifications
   to any of its public licenses or any other arrangements,
   understandings, or agreements concerning use of licensed material. For
   the avoidance of doubt, this paragraph does not form part of the
   public licenses.

   Creative Commons may be contacted at creativecommons.org.
*/
