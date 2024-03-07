/*
 * star_kernel.h:
 *  элементы языка в __device__-функциях с заданием кол-ва блоков и потоков
 *  проблемы с GetRow, SetRow, FND по числу блоков-нитей.
 *
 */
#include "param.h"
#include "find.h"
#define index blockIdx.x
//threadIdx.x + blockIdx.x*blockDim.x

      extern  const int NN1;
      extern  int threads1;
	  extern int blocks1;

/*
 * Переменные фиксированы для вызова GetRow, SetRow
 * FND <<<1,N>>> и <<<1,1>>> критично к одновременному окончанию всех нитей
 * Для остальных функций threads=1; остальными нитями можно играть в цикле while SOME(X)
 */
//int  threads= M < SIZE_OF_LONG_INT ? M : SIZE_OF_LONG_INT; //64, если больше 64 вершин в графе, иначе кол-во вершин.
//int  blocks= (LENGTH1 > SIZE_OF_LONG_INT)? (int)ceil( M/(double)threads) : 1 ; // кол-во 64-х разрядных чисел, необходимых для записи столбца/строки.

	  void fusedDscal();
//SLICES

__device__ void _and(unsigned long long int *d_v,unsigned long long int *d_v1);
//__device__ void _and(unsigned long long int *d_v,unsigned long long int v1);
//__device__ void _and(unsigned long long int v,unsigned long long int *d_v1);
//__device__ void _and(unsigned long long int v,unsigned long long int v1);

__device__ void _not(unsigned long long int *d_v);
//__device__ void _not(unsigned long long int v);

__device__ void _or(unsigned long long int *d_v,unsigned long long int *d_v1);
//__device__ void _or(unsigned long long int *d_v,unsigned long long int v1);
//__device__ void _or(unsigned long long int v,unsigned long long int *d_v1);
//__device__ void _or(unsigned long long int v,unsigned long long int v1);

__device__ void _xor(unsigned long long int *d_v,unsigned long long int *d_v1);
//__device__ void _xor(unsigned long long int *d_v,unsigned long long int v1);
//__device__ void _xor(unsigned long long int v,unsigned long long int *d_v1);
//__device__ void _xor(unsigned long long int v,unsigned long long int v1);

__device__ int _get_bit(unsigned long long int *d_v,int k);
__device__ void _mask(unsigned long long int *d_v, int num);
__device__ LongPointer _col(LongPointer *d_tab,int i);
__device__ void _clr(unsigned long long int *d_v);
//__device__ void _clr(unsigned long long int v);
__device__ void _set(unsigned long long int *d_v);
//__device__ void _set(unsigned long long int v);
__device__ void _assign(unsigned long long int *d_v,unsigned long long int *d_u);//d_v[]=d_u[] global slices
//__device__ void _assign(unsigned long long int v,unsigned long long int *d_u);//v=d_u[] v-local element of slice
__device__ void _assign(unsigned long long int *d_v,unsigned long long int u);//d_v[]=u u-local element of slice
//__device__ void _assign(unsigned long long int v,unsigned long long int u); //u,v - local elements of slices
__device__ unsigned long long int _assign(unsigned long long int *d_v);
