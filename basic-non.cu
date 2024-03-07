#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/copy.h>
#include <algorithm>
#include <time.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

#include <thrust/detail/config.h>

//#include <thrust/system/cuda/detail/cuda_launch_config.h>
#include <thrust/tuple.h>

#include "param.h"

#define LOOPS 100
using namespace thrust::placeholders;

inline void find_min_max(thrust::device_vector<int> &dev_vec, int *min, int *max){
    thrust::pair<thrust::device_vector<int>::iterator,thrust::device_vector<int>::iterator> tuple;
    tuple = thrust::minmax_element(dev_vec.begin(),dev_vec.end());
    *min = *(tuple.first);
    *max = *tuple.second;
}
inline void find_min(thrust::device_vector<int> &dev_vec, int *min){
    thrust::device_vector<int>::iterator iter;
 //   thrust::detail::execution_policy_base<DerivedPolicy> &exec;

    iter = thrust::min_element(dev_vec.begin(),dev_vec.end());
    *min = *iter;

}
inline void find_min1(thrust::device_vector<int> &dev_vec, int *min){
    thrust::device_vector<int>::iterator iter;

    iter = thrust::min_element(thrust::cuda::par,dev_vec.begin(),dev_vec.end());
    *min = *iter;

}

class min_pos
{ int val;
public:
__host__ __device__ min_pos(int min){val=min;}
__host__ __device__ int operator()(int & c)const {return (c==val)?1:0;}
};

struct is_even
  {
    __host__ __device__
    bool operator()(const int x)
    {
      return (x % 2) == 0;
    }
};

struct equal_to
  {
    __host__ __device__
    bool operator()(const int x, const int y)
    {
      return (x == y);
    }
};
/*
class minus_pred
{ bool val;
public:
__host__ __device__ minus_pred(class Pred p){val=p();}
__host__ __device__ int operator()(int & c1,int & c2)const {return val?(c1-c2):0;}
};*/

template<class In, class Out, class Pred>
Out copy_if(In first, In last, Out res,Pred p)
{
 while (first!=last)
 {if (p(*first)) *res++=*first;
 ++first;}
 return res;
}
template<class In, class Out, class Pred>
void minus_if(In first1, In last1, In first2, Out res,Pred p)
{
 while (first1!=last1)
 {if (p(*first1)) *res++=*first1-*first2;
 ++first1;
 ++first2;}
 return res;
}

int maintest(){
    int minele;//, maxele;
    char fout[50];
         sprintf(fout,"res/test_lib/res_%i_thrust.dat",LENGTH1);
    FILE *f_out;
    if((f_out=fopen(fout,"wt"))==NULL)return 0;

    int N=LENGTH1;
    std::vector<int> a,b(N),res(N);
    for (int i=0; i<N; i++)
    {
      a.push_back(rand());
      b[i]=a[i];
    }
    int j=N/LOOPS;
    thrust::host_vector<int> h_a(N),h_b(N);
    thrust::copy(a.begin(), a.end(), h_a.begin());
    thrust::device_vector<int> d_a = h_a;
    thrust::device_vector<int> d_b=d_a,d_res(N);
    thrust::device_vector<int>::iterator iter;
 //   thrust::copy(d_a.begin(),d_a.begin(),d_b.begin());

    cudaEvent_t start, stop;
//=========================================================================================
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int i=0; i < LOOPS; i++){
  // ------MINUS_IF-------------------------
    	thrust::copy(d_a.begin(),d_a.end(),d_b.begin());
  //  	thrust::transform( d_a.begin(),d_a.end(),d_b.begin(),d_res.begin(),thrust::minus<int>());
    	 thrust::transform_if(d_a.begin(),d_a.end(),d_b.begin(),d_a.begin(), d_res.begin(),thrust::minus<int>(),_1%2==0);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime1, totalTime1;
    cudaEventElapsedTime(&elapsedTime1, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    totalTime1 = elapsedTime1/(1000*LOOPS);
    cudaError_t err = cudaGetLastError();
// 	printf("errors after thrust %d\n",err);
    fprintf(f_out,"thrust minus_if time = %f\n", totalTime1);

    //========================================================================================
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int i=0; i < LOOPS; i++){
// ------MINIMUM_IF-------------------------
    	  thrust::device_vector<int> d_a = h_a;
        thrust::copy(h_a.begin(), h_a.end(), d_a.begin());

      find_min(d_a,&minele);
     thrust::device_vector<int> d_b(N);
      thrust::fill(d_b.begin(),d_b.end(),0);
//      thrust::copy(d_a.begin(),d_a.begin(),d_b.begin());
      thrust::transform(d_a.begin(),d_a.end(),d_b.begin(),min_pos(minele));// позиции минимальных элементов
//      thrust::transform_if(d_a.begin(),d_a.end(),d_b.begin(),d_a.begin(), d_res.begin(),min_pos(minele),_1%2==0);
//     thrust::transform(thrust::cuda::par, d_a.begin(),d_a.end(),d_b.begin(),d_res.begin(),thrust::minus<int>());
//       thrust::stable_sort(d_a.begin(),d_a.end(),thrust::greater<int>());
 //   	thrust::copy_if(d_a.begin(),d_a.end(),d_b.begin(),is_even());
 //       thrust::transform_if(d_a.begin(),d_a.end(),d_b.begin(),d_a.begin(), d_res.begin(),thrust::plus<int>(),_1%2==0);
      }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
 //   float elapsedTime1, totalTime1;
    cudaEventElapsedTime(&elapsedTime1, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    totalTime1 = elapsedTime1/(1000*LOOPS);
//    cudaError_t
    err = cudaGetLastError();
 //	printf("errors after thrust %d\n",err);

//    printf("thrust min element = %d\n", minele);
    fprintf(f_out,"thrust min time = %f\n", totalTime1);        //

    //=========================================================================================
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    for (int i=0; i < LOOPS; i++){
// ------MATCH-------------------------
    	  thrust::device_vector<int> d_a = h_a;
        thrust::copy(h_a.begin(), h_a.end(), d_a.begin());

//      find_min(d_a,&minele);
     thrust::device_vector<int> d_b(N);
      thrust::fill(d_b.begin(),d_b.end(),0);
//      thrust::copy(d_a.begin(),d_a.begin(),d_b.begin());
 //     iter = thrust::find_if(a.begin(), a.end(),[i](const int x){ return x == i; } );
      thrust::transform(d_a.begin(),d_a.end(),d_b.begin(),min_pos(i));

//     thrust::transform(thrust::cuda::par, d_a.begin(),d_a.end(),d_b.begin(),d_res.begin(),thrust::minus<int>());
//       thrust::stable_sort(d_a.begin(),d_a.end(),thrust::greater<int>());
 //   	thrust::copy_if(d_a.begin(),d_a.end(),d_b.begin(),is_even());
 //       thrust::transform_if(d_a.begin(),d_a.end(),d_b.begin(),d_a.begin(), d_res.begin(),thrust::plus<int>(),_1%2==0);
      }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
 //   float elapsedTime1, totalTime1;
    cudaEventElapsedTime(&elapsedTime1, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    totalTime1 = elapsedTime1/(1000*LOOPS);
//    cudaError_t
    err = cudaGetLastError();
 //	printf("errors after thrust %d\n",err);

//    printf("thrust min element = %d\n", minele);
    fprintf(f_out,"thrust match_if time = %f\n", totalTime1);        //

//=========================================================================================
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i=0; i < LOOPS; i++){
// ------COPY_IF-------------------------
    	  thrust::device_vector<int> d_a = h_a;
//        thrust::copy(h_a.begin(), h_a.end(), d_a.begin());
     thrust::device_vector<int> d_b(N);
    	thrust::copy_if(d_a.begin(),d_a.end(),d_b.begin(),is_even());
 //       thrust::transform_if(d_a.begin(),d_a.end(),d_b.begin(),d_a.begin(), d_res.begin(),thrust::plus<int>(),_1%2==0);
      }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
//    float elapsedTime1, totalTime1;
    cudaEventElapsedTime(&elapsedTime1, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    totalTime1 = elapsedTime1/(1000*LOOPS);
    //cudaError_t
    err = cudaGetLastError();
 //	printf("errors after thrust %d\n",err);

//    printf("thrust min element = %d\n", minele);
    fprintf(f_out,"thrust copy_if time = %f\n", totalTime1);        //

//    thrust::copy(d_b.begin(),d_b.end(),b.begin());
     clock_t t;

     std::vector<int>::iterator resultmax, resultmin;

 //=============================================================
     t = clock();
/*    std::sort(a.begin(), a.end());
    t = clock() - t;
    printf("STL sort time = %f\n", ((float)t)/(CLOCKS_PER_SEC));
*/

    for (int i = 0; i<LOOPS; i++){
 //     resultmax = std::max_element(a.begin(), a.end());
     resultmin = std::min_element(a.begin(), a.end());

 //   	std::transform(a.begin(),a.end(),b.begin(),res.begin(),std::minus<int>());
  //  	resultmax=std::lower_bound(a.begin(), a.end(),b[N-j*i]);
    //	std::copy(a.begin(),a.end(),b.begin());
 //   	minus_if(a.begin(),a.end(),b.begin(),res.begin(),is_even());
 //   	std::transform_if(a.begin(),a.end(),b.begin(),a.begin(), res.begin(),std::minus<int>(),_1%2==0);
     }
    t = clock() - t;
//    resultmax=resultmin;
 //   printf("STL sort= %d, max element = %d\n", b[j], *resultmax);
    fprintf(f_out,"STL min time = %f\n", ((float)t)/(CLOCKS_PER_SEC*LOOPS));
//====================================================================================
     t = clock();
     //copy_if(a.begin(),a.end(),b.begin(),is_even());
     for (int i = 0; i<LOOPS; i++) std::copy_if(a.begin(),a.end(),b.begin(),is_even());

     t = clock() - t;
     //    resultmax=resultmin;
      //   printf("STL sort= %d, max element = %d\n", b[j], *resultmax);
         fprintf(f_out,"STL copy_if time = %f\n", ((float)t)/(CLOCKS_PER_SEC*LOOPS));

          t = clock();
    	 std::transform(a.begin(),a.end(),b.begin(),res.begin(),std::minus<int>());
     t = clock() - t;
     fprintf(f_out,"STL minus time = %f\n", ((float)t)/(CLOCKS_PER_SEC));//*LOOPS));*/
 //    puts("d_b");
 //   thrust::copy(d_res.begin(), d_res.end(), std::ostream_iterator<int>(std::cout, "\n"));
//    thrust::copy(d_res.begin(),d_res.end(),res.begin());
//======================================================================================
     t = clock();
      //copy_if(a.begin(),a.end(),b.begin(),is_even());
      for (int i = 0; i<LOOPS; i++)
     	 std::find(a.begin(),a.end(),10);
     	 //transform(a.begin(),a.end(),b.begin(),res.begin(),std::minus<int>());
      t = clock() - t;
      fprintf(f_out,"STL match time = %f\n", ((float)t)/(CLOCKS_PER_SEC));//*LOOPS));*/
// for(int i=0;i<N;i++) printf("a,b[%i]=%i res=%i\n",i,b[i],res[i]);

//    thrust::copy(d_a.begin(), d_a.end(), std::ostream_iterator<int>(std::cout, "\n"));
  return 0;
}
