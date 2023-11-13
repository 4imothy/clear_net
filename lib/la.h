#ifndef CN_LA
#define CN_LA

#include "clear_net.h"

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

#endif // CN_LA
