#ifndef CN_LA
#define CN_LA

#include "clear_net.h"
#include <stdio.h>

typedef enum {
    VECTORS,
    MATRICES,
    MULTIMATRICES,
} DataType;

typedef union {
    Vector *vectors;
    Matrix *matrices;
    Matrix **multi_matrices;
} LAData;

struct CNData {
    DataType type;
    ulong nchannels;
    ulong nelem;
    LAData in;
};

Matrix formMatrix(ulong nrows, ulong ncols, ulong stride, scalar *elements);
Matrix allocMatrix(ulong nrows, ulong ncols);
void deallocMatrix(Matrix *mat);
void printMatrix(Matrix *mat, char *name);
void shuffleDatas(CNData *input, CNData *target);
Vector formVector(ulong nelem, scalar *elem);
Vector allocVector(ulong nelem);
void deallocVector(Vector *vec);
void printVector(Vector *vec, char *name);
void printVectorInline(Vector *vec);
void setBatchFromMatrix(Matrix all_input, Matrix all_target, ulong batch_num,
                        ulong batch_size, Matrix *batch_in, Matrix *batch_tar);
CNData *allocDataFromVectors(Vector *vectors, ulong nelem);
CNData *allocDataFromMatrices(Matrix *matrices, ulong nelem);
CNData *allocDataFromMultiChannelMatrices(Matrix **multi_matrices, ulong nelem,
                                          ulong nchannels);
CNData *allocEmptyData(void);
void deallocData(CNData *data);
void printData(CNData *d);
Vector *allocVectors(ulong count, ulong nelem);
Matrix *allocMatrices(ulong count, ulong nrows, ulong ncols);
Matrix **allocMultiMatrices(ulong count, ulong nchannels, ulong nrows,
                            ulong ncols);
void deallocVectors(Vector *list, ulong count);
void printVectors(Vector *list, ulong count);
void deallocMatrices(Matrix *list, ulong count);
void printMatrices(Matrix *list, ulong count);
void deallocMultiMatrices(Matrix **list, ulong count, ulong nchannels);
void printMultiMatrices(Matrix **list, ulong count, ulong nchannels);
void setBatch(CNData *all_input, CNData *all_target, ulong batch_num,
              ulong batch_size, CNData *batch_in, CNData *batch_tar);

#endif // CN_LA
