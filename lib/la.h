#ifndef CN_LA
#define CN_LA

#include "clear_net.h"

typedef enum {
    VecList,
    MatList,
    MultiMatList,
} IODataType;

typedef union {
    Vector* vec_list;
    Matrix* mat_list;
    Matrix** multi_mat_list;
} LAData;

struct IOData {
    IODataType type;
    ulong nchannels;
    ulong nelem;
    LAData data;
};

Matrix formMatrix(ulong nrows, ulong ncols, ulong stride, scalar *elements);
Matrix allocMatrix(ulong nrows, ulong ncols);
void deallocMatrix(Matrix *mat);
void printMatrix(Matrix *mat, char *name);
void shuffleMatrixRows(Matrix *input, Matrix *target);
Vector formVector(ulong nelem, scalar *elem);
Vector allocVector(ulong nelem);
void deallocVector(Vector *vec);
void printVector(Vector *vec, char *name);
void printVectorInline(Vector *vec);
void setBatchFromMatrix(Matrix all_input, Matrix all_target, ulong batch_num,
                        ulong batch_size, Matrix *batch_in, Matrix *batch_tar);
IOData *formDataFromVectors(Vector *vectors, ulong nelem);
IOData *formDataFromMatrices(Matrix *matrices, ulong nelem);
IOData *formDataFromMultiChannelMatrices(Matrix **multi_matrices, ulong nelem, ulong nchannels);

#endif // CN_LA
