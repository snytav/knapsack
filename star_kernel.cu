#include "star_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#define col_bits 128
//#define c_th 100
        const int NN1=(LENGTH1-1)/64+1;
        int threads1=1;//(col_bits<NN1)?col_bits:NN1;
		int blocks1=NN1;//(NN1-1)/threads1+1;  //?15 одномоментно?
//#include "find.h"
//#define index  threadIdx.x + blockIdx.x*blockDim.x

//const int NN=(LENGTH1-1)/SIZE_OF_LONG_INT+1;

		// Fused the diagonal element root and dscal operation into
		// a single "cdiv" operation
		void launchMyKernel(int *array, int arrayCount)
		{
		  int blockSize;   // The launch configurator returned block size
		  int minGridSize; // The minimum grid size needed to achieve the
		                   // maximum occupancy for a full device launch
		  int gridSize;    // The actual grid size needed, based on input size

//		  cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
//		                                      MyKernel, 0, 0);
		  // Round up according to array size
		  gridSize = (arrayCount + blockSize - 1) / blockSize;

//		  MyKernel<<< gridSize, blockSize >>>(array, arrayCount);

		  cudaDeviceSynchronize();

		  // calculate theoretical occupancy
		  int maxActiveBlocks;
//		  cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks,
//		                                                 MyKernel, blockSize,
//	                                                 0);

		  int device;
		  cudaDeviceProp props;
		  cudaGetDevice(&device);
		  cudaGetDeviceProperties(&props, device);

		  float occupancy = (maxActiveBlocks * blockSize / props.warpSize) /
		                    (float)(props.maxThreadsPerMultiProcessor /
		                            props.warpSize);

		  printf("Launched blocks of size %d. Theoretical occupancy: %f\n",
		         blockSize, occupancy);
		}


		void fusedDscal()
		{
		    // The semibandwidth (column length) determines
		    // how many warps are required per column of the
		    // matrix.
			cudaDeviceProp devProp;
			cudaGetDeviceProperties ( &devProp, 0);

			int n=NN1;
		    const int warpSize = 32;
		    const int maxGridSize =14;//112; // this is 8 blocks per MP for a Telsa C2050

		    int warpCount = ((n -1)/ warpSize+1);// + (((n % warpSize) == 0) ? 0 : 1);
		    int warpPerBlock = max(1, min(4, warpCount));
		    // For the cdiv kernel, the block size is allowed to grow to
		    // four warps per block, and the block count becomes the warp count over four
		    // or the GPU "fill" whichever is smaller
		    int threadCount = warpSize * warpPerBlock;
		    int blockCount = min( maxGridSize, max(1, (warpCount-1)/warpPerBlock+1) );
//
            int warpInstruction=(NN1-1)/(threadCount*blockCount)+1;
//
		    dim3 BlockDim = dim3(threadCount, 1, 1);
		    dim3 GridDim  = dim3(blockCount, 1, 1);
        printf("calc: blocks=%i, threads=%i ,InsPerThread=%i\n",blockCount,threadCount,warpInstruction);

         printf ( "Compute capability : %d.%d\n", devProp.major, devProp.minor );
         printf ( "Name : %s\n", devProp.name );
         printf ( "Total Global Memory : %d\n", devProp.totalGlobalMem );
         printf ( "Shared memory per block: %d\n", devProp.sharedMemPerBlock );
         printf ( "Registers per block : %d\n", devProp.regsPerBlock );
         printf ( "Warp size : %d\n", devProp.warpSize );
         printf ( "Max threads per block : %d\n", devProp.maxThreadsPerBlock );
         printf ( "Total constant memory : %d\n", devProp.totalConstMem );
         printf("Multiprocessor count: %d\n", devProp.multiProcessorCount);

 //		void launchMyKernel(int *array, NN);

		}
//////////////////////////////////////////////////////////////////////////////////////////

__device__ void _and(unsigned long long int *d_v,unsigned long long int *d_v1)
		{

	if (index<NN1) d_v[index] &= d_v1[index];
		};

////////////////////////////////////////////////
__device__ void _or(unsigned long long int *d_v,unsigned long long int *d_v1)
		{

	if (index<NN1) d_v[index] |= d_v1[index];
		};

////////////////////////////////////////////////
__device__ void _xor(unsigned long long int *d_v,unsigned long long int *d_v1)
		{
	if (index<NN1) d_v[index] ^= d_v1[index];
		};

//////////////////////////////////////////////////

__device__ void _not(unsigned long long int *d_v)
		{
	if (index<NN1) d_v[index] = ~d_v[index];
		};

__device__ int _get_bit(unsigned long long int *d_v,int k)
{
	int num,sh;
	unsigned long long int p = 1;

	num = get_64bit_word(k,SIZE_OF_LONG_INT);
	sh =  position_in_64bit_word(k,SIZE_OF_LONG_INT);
	p = p << (sh-1);
#ifdef bbb
	printf("get_positio_bit n %d num %d sh %d shifted p %llu \n",k,num,d_v[num],p);
#endif
	return (d_v[num] & p ) && 1;
}

__device__ void _mask(unsigned long long int *d_v, int num)
{ unsigned long long int zero;
  int num_el=num>>6; // номер элемента, содержащий переход от 0 к 1;

	  if (index==num_el)
   {
	  zero=1>>(num % SIZE_OF_LONG_INT)-1;
   }
  else
  {
      zero=0;
      if (index>num_el)
      {
    	  zero=!zero;
      }
  }
   d_v[index]=zero;
}

__device__ LongPointer _col(LongPointer *d_tab,int i)
{
	return d_tab[i];
}

__device__ void _clr(unsigned long long int *d_v)
{
	if (index<NN1) d_v[index] = 0;
}

__device__ void _set(unsigned long long int *d_v)
{ unsigned long long int zero = 0;

	if (index<NN1) d_v[index]= ~zero;
}

__device__ void _assign(unsigned long long int *d_v,unsigned long long int *d_u)
{
	if (index<NN1)d_v[index]=d_u[index];
}

__device__ void _assign(unsigned long long int *d_v,unsigned long long int u)
{
	if (index<NN1) d_v[index]=u;
}

__device__ unsigned long long int _assign(unsigned long long int *d_u)
{
	if (index<NN1) return d_u[index];
	else return 0;
}


