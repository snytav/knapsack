#include "starmachine.h"
#include "basicadj.h"

__device__ void match(LongPointer T, bool *X, LongPointer w, bool *Y)
{
	if((index)<edge)
	{	Y[index]=X[index];
       bool *col;
	    for(int i=0; i<vertex;i++)
	    {
	       col=T[i];
	       Y[index]=Y[index]&&(col[index]==w[i]);
	    }
	}
//__syncthreads();
}
