#include <stdlib.h> // for NULL
#include "cJADE.h"


/* .C calls */
extern void FG(double *X, double *b, int *kpmaxit, double *w, double *eps, double *result);
extern void rjdc(double *X, int *kpmaxit, double *w, double *eps, double *result);

