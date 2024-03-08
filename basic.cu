
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
//#include <sys/time.h>
#include <cuda.h>
#include "basic.h"
#include "star_kernel.h"
//#include "cuPrintf.cu"


void MATCH(Table *tab, Slice *X, Slice *w, Slice *Z)
{

	unsigned int j,k;
	int s;
	k=tab->size;
//printf("\n MATCH %i \n", k);
	j=X->length;
 Slice *Y;
 Y=new Slice;
 Y->Init(j);

 Z->assign(X);
 for(int i=1;i<=k;i++)
 {
 tab->GetCol(Y,i);
  s= w->get(i);
  if (s==1) Z->AND(Y);
  else
  {
	  Y->NOT();
	  Z->AND(Y);
  }
 }
}

__device__ void match(LongPointer *d_tab,unsigned long long int *d_x,unsigned long long int *d_w,unsigned long long int *d_z,int size)
{	 unsigned long long int *d_col=new unsigned long long int[gridDim.x];
	 int s;
//	 int index=blockIdx.x; //threadIdx.x + blockIdx.x*blockDim.x;
//	 cuPrintf("march_kernel \n");
	 _assign(d_z,d_x);

	 for(int i=0;i<size;i++)
	  {
//	  d_d=(unsigned long long *)(d_tab[i]);
//		 d_d=d_tab[i];
//	  d_col=d_d[blockIdx.x];
	  _assign(d_col,_col(d_tab,i));//d_tab[i];
	  s=get_position_bit(d_w,i+1);// s= d_w->get(i);
//	  printf("i =%d blockIdx.x= %d d_z[]= %ull d_col %ull \n",i,blockIdx.x,d_w[0],d_col);
 	   if (s==1)
	   {
	 	  //Z->AND(Y);
	 	 _and(d_z,d_col);
	    }
	   else
	   {
	 	  _not(d_col);
//	 	 printf("i =%d blockIdx.x= %d not d_col %ull \n",i,blockIdx.x,d_col);
	 	  //Y->NOT();
	 	//  Z->AND(Y);
	 	 _and(d_z,d_col);
	   }
  // printf("i =%d blockIdx.x= %d s=%d res= %ull \n",i,blockIdx.x,s,d_z[blockIdx.x]);
	  }
	 delete [] d_col;
	 }

 __global__ void match_kernel(LongPointer *d_tab,unsigned long long int *d_x,unsigned long long int *d_w,unsigned long long int *d_z,int size)
 {
    match(d_tab,d_x,d_w,d_z,size);
	 }

 void MATCH_CUDA(Table *tab, Slice *X, Slice *w, Slice *Z)
{
	 unsigned long long int *d_x,*d_z, *d_w;
	 LongPointer *d_tab;
     int Nl=tab->size;
     int NN=X->NN;

     d_x=X->get_device_pointer();
	 d_w=w->get_device_pointer();
	 d_z=Z->get_device_pointer();
	 d_tab=tab->get_device_pointer();

	 match_kernel<<<NN,1>>>(d_tab,d_x,d_w,d_z,Nl);
 }
 __device__ void gel(LongPointer *d_tab, unsigned long long int * d_w,unsigned long long int *d_x,unsigned long long int *d_y,int size)
 {     unsigned long long int x,y,z,b,*d_col;
       int s;

	    x=0;//_clr(d_x);
	    y=0;//_clr(d_y);
        z=~0;//_set(z);
       for(int i=0;i<size;i++)
       {
    	 d_col=_col(d_tab,i);
    	 b=_assign(d_col);
         s=get_position_bit(d_w,i+1);
		 if (s==1)
		 {
//	(* In the slice Y we accumulate position of those i-th rows for which row(i,T)<w. *)
		   b=(~b) &z;//not(b); _and(b,z);
		   y|=b;     //_or(d_y,b);
//	printf("%i:l=%llu \n",blockIdx.x,b);
		 }
		 else{
			// _and(b,z);
			x|=b&z;// _or(d_x,b);
//	printf("%i:g=%llu \n",blockIdx.x,b);
//	(* In the slice X we accumulate position of those i-th rows for which row(i,T)>w. *)
		 }
		 //_not(b);
		 z&=~b;//_and(z,b);
//	(* Positions of the selected rows are deleting from the slice Z. *)
       }
       _assign(d_x,x);
       _assign(d_y,y);
 }
__global__ void gel_kernel(LongPointer *d_tab, unsigned long long int * d_w,unsigned long long int *d_x,unsigned long long int *d_y,int size)
{
    gel(d_tab, d_w, d_x,d_y,size);
}
 void GEL(Table *T, Slice *w, Slice *X,Slice *Y){
 unsigned long long int *d_x,*d_y, *d_w;
	 	 LongPointer *d_tab;
	      int NN=X->NN;
	      int size=T->size;
	      d_x=X->get_device_pointer();
	 	 d_w=w->get_device_pointer();
	 	 d_y=Y->get_device_pointer();
	 	 d_tab=T->get_device_pointer();
	 	gel_kernel<<<NN,1>>>(d_tab,d_w,d_x,d_y,size);
 }


void LESS(Table *T, Slice *X, Slice *v,Slice *Y)
{ unsigned long long int *d_x,*d_y, *d_v;
 LongPointer *d_tab;
 int NN=X->NN;
 int size=T->size;
 d_x=X->get_device_pointer();
 d_v=v->get_device_pointer();
 d_y=Y->get_device_pointer();
 d_tab=T->get_device_pointer();
  less_kernel<<<NN,1>>>(d_tab,d_x,d_v,d_y,size);
};

__global__ void less_kernel(LongPointer *d_tab,unsigned long long int *d_x,unsigned long long int *d_v,unsigned long long int *d_y, int size)
{
   less(d_tab,d_x,d_v,d_y,size);
};

__device__ void less(LongPointer *d_tab,unsigned long long int *d_x,unsigned long long int *d_v,unsigned long long int *d_y, int size)
{
	unsigned long long int b,y,y1, *d_col;
	int s;

   y1=_assign(d_x);
   y=0;
   for(int i=0;i<size;i++)
   {
      d_col=_col(d_tab,i);
	  b=_assign(d_col);
	  s=get_position_bit(d_v,i+1);
	  if (s==1)
	  {
	 	b=y1&(~b);
	 	y|=b;
	  }
	  y1&=~b;
    }
   _assign(d_y,y);
};

//procedure Great(T: Table; X: slice; v: word; Var Y: slice);
//Var
//  B, C: Slice;
//  n, i: integer;
//begin
//     B:=X;
//     Clr(Y);
//     For i:=1 to n do
//     begin
//      C:=col(i, T);
//      if v(i)=0 then
//      begin
 //        C:=C and B;
 //        Y:=C xor Y;
 //        C:=not C;
 //     end;
 //     B:=C and B;
 //    end;
//end;
void GREAT(Table *T, Slice *X, Slice *v,Slice *Y)
{ unsigned long long int *d_x,*d_y, *d_v;
 LongPointer *d_tab;
 int NN=X->NN;
 int size=T->size;
 d_x=X->get_device_pointer();
 d_v=v->get_device_pointer();
 d_y=Y->get_device_pointer();
 d_tab=T->get_device_pointer();
  less_kernel<<<NN,1>>>(d_tab,d_x,d_v,d_y, size);
};

__global__ void great_kernel(LongPointer *d_tab,unsigned long long int *d_x,unsigned long long int *d_v,unsigned long long int *d_y, int size)
{
   great(d_tab,d_x,d_v,d_y, size);
};

__device__ void great(LongPointer *d_tab,unsigned long long int *d_x,unsigned long long int *d_v,unsigned long long int *d_y,int size)
{
	unsigned long long int b,c,y, *d_col;
	int s;

   b=_assign(d_x);
   y=0;//_clr(d_y);
   for(int i=0;i<size;i++)
   {
      d_col=_col(d_tab,i);
	  c=_assign(d_col);
	  s=get_position_bit(d_v,i+1);
	  if (s==0)
	  {
	 	c&=b;//_and(c,b);
	 	y^=c;//_xor(d_y,c);
	 	c=~c;//_not(c);
	  }
	  b&=c;//_and(b,c);

    }
   _assign(d_y,y);
};
__global__ void min_part(LongPointer *d_tab,unsigned long long *d_y,int i,unsigned long long int *d_z)//<<<NN,1>>>
{
	unsigned long long int *d_col;
      d_col=_col(d_tab,i);
	  _assign(d_y,d_col);
	  _not(d_y);
	  _and(d_y,d_z);
};

__global__ void min_part1(LongPointer *d_tab,unsigned long long *d_y,int i,unsigned long long int *d_z,int *d_first_non_zero,int size)//<<<NN,1>>>
{
	unsigned long long int *d_col;
//	printf("MIN_part1     i=%i first=%i \n",i,*d_first_non_zero);
	 if ((i>0)&(d_first_non_zero[0]>0))
		 d_z[blockIdx.x]=d_y[blockIdx.x];//_assign(d_z,d_y);
	 if (i<size)
	 {
      d_col=_col(d_tab,i);
	 // _assign(d_y,d_col);
      d_y[blockIdx.x]=~d_col[blockIdx.x];
	 // _not(d_y);
	 // _and(d_y,d_z);
      d_y[blockIdx.x]&=d_z[blockIdx.x];
	 }
};

void MIN(Table *T, Slice *X, Slice*Z)
{unsigned long long int *d_z,*d_y;
int *d_first_non_zero;
LongPointer *d_t;
int size=T->size;
// cudaError_t err = cudaGetLastError();
// printf("before MIN %d , %s \n",err,cudaGetErrorString(err));
 Slice *Y;
 Y=new Slice;
 Y->Init(LENGTH1);

 d_y=Y->get_device_pointer();
 d_z=Z->get_device_pointer();
 d_t=T->get_device_pointer();
 Z->assign(X);
// err = cudaGetLastError();
//  printf("after init %d , %s \n",err,cudaGetErrorString(err));
 cudaMalloc(&d_first_non_zero,sizeof(int));

 int i=0;
// err = cudaGetLastError();
//  printf("after cudaMalloc %d , %s \n",err,cudaGetErrorString(err));


 // Z->print("MIN:Z",0);
 
 //(LongPointer * d_tab, unsigned long long* d_y, int i, unsigned long long int* d_z, int* d_first_non_zero, int size)//<<<NN,1>>>

 min_part1<<<NN1,1>>>(d_t,d_y,i,d_z,d_first_non_zero,size);
// Y->print("MIN:Y",0);
// Z->print("MIN:Z",0);
 for(i=1;i<=size;i++)
{
	 first(d_y,NN1,d_first_non_zero,NN1);

     //min_part1<<<blocks1,threads1>>>(d_t,d_y,i,d_z,d_first_non_zero);
	 min_part1<<<NN1,1>>>(d_t,d_y,i,d_z,d_first_non_zero,size);
//err = cudaGetLastError();
//printf("%i MIN %d , %s \n",i,err,cudaGetErrorString(err));
//	 Y->print("MIN:Y",0);
 //    Z->print("MIN:Z",0);
 }
};

void MIN(Table *T, Slice *X, Slice*Z, Slice *Y)
{unsigned long long int *d_z,*d_y;
  int *d_first_non_zero;
  int size=T->size;
LongPointer *d_t;
 d_y=Y->get_device_pointer();
 d_z=Z->get_device_pointer();
 d_t=T->get_device_pointer();
 Z->assign(X);
 cudaMalloc(&d_first_non_zero,sizeof(int));
 int i=0;
 min_part1<<<blocks1,threads1>>>(d_t,d_y,i,d_z,d_first_non_zero,size);
 for(i=1;i<size;i++)
{
	 first(d_y,NN1,d_first_non_zero,NN1);
     min_part1<<<blocks1,threads1>>>(d_t,d_y,i,d_z,d_first_non_zero,size);
 }
};

void MIN_1(Table *T, Slice *X, Slice*Z, Slice *Y)
{unsigned long long int *d_z,*d_y;
  int *d_first_non_zero;
  int size=T->size;
LongPointer *d_t;
 d_y=Y->get_device_pointer();
 d_z=Z->get_device_pointer();
 d_t=T->get_device_pointer();
 Z->assign(X);
 cudaMalloc(&d_first_non_zero,sizeof(int));
 int i=0;
 min_part1<<<blocks1,threads1>>>(d_t,d_y,i,d_z,d_first_non_zero);
 for(i=1;i<size;i++)
{
	 some(d_y,NN1,d_first_non_zero,NN1);
     min_part1<<<blocks1,threads1>>>(d_t,d_y,i,d_z,d_first_non_zero);
 }
};
__global__ void max_part(LongPointer *d_tab,unsigned long long *d_y,int i,unsigned long long int *d_z)//<<<NN,1>>>
{
	unsigned long long int *d_col;
//	 if (k>0) _assign(d_z,d_y);
      d_col=_col(d_tab,i);
	  _assign(d_y,d_col);
	  _and(d_y,d_z);
};
__global__ void max_part(LongPointer *d_tab,unsigned long long *d_y,int i,unsigned long long int *d_z,int *d_first_non_zero)//<<<NN,1>>>
{
	unsigned long long int *d_col;
//	 if (k>0) _assign(d_z,d_y);
	 if ((i>0)&(d_first_non_zero>0)) _assign(d_z,d_y);
      d_col=_col(d_tab,i);
	  _assign(d_y,d_col);
	  _and(d_y,d_z);
};
void MAX(Table *T, Slice *X, Slice*Z)
{unsigned long long int *d_z,*d_y;
int *d_first_non_zero;
LongPointer *d_t;
int NN=X->NN;
int size=T->size;
 Slice *Y;
 Y=new Slice;
 Y->Init(LENGTH1);

 d_y=Y->get_device_pointer();
 d_z=Z->get_device_pointer();
 d_t=T->get_device_pointer();
 Z->assign(X);
 cudaMalloc(&d_first_non_zero,sizeof(int));

 cudaError_t err = cudaGetLastError();
 printf("before MAX %d , %s \n",err,cudaGetErrorString(err));

 int i=0;
  max_part<<<blocks1,threads1>>>(d_t,d_y,i,d_z,d_first_non_zero);
  err = cudaGetLastError();
  printf("first MAX %d , %s \n",err,cudaGetErrorString(err));
  for(i=1;i<size;i++)
 {
 	 first(d_y,NN1,d_first_non_zero,NN1);
 	 puts("first");
      max_part<<<blocks1,threads1>>>(d_t,d_y,i,d_z,d_first_non_zero);
      printf("%d ",i);
  }
  err = cudaGetLastError();
  printf("after MAX %d , %s \n",err,cudaGetErrorString(err));
};

//__device__ void min(LongPointer *d_tab,unsigned long long int *d_x,unsigned long long int *d_z)// нет из-за синхронизации перед some()

void SETMIN(Table *T, Table *F, Slice *X, Slice *Z)
{unsigned long long int *d_x,*d_z;
LongPointer *d_t, *d_f;
int NN=X->NN;
int size=T->size;
d_x=X->get_device_pointer();
d_z=Z->get_device_pointer();
d_t=T->get_device_pointer();
d_f=F->get_device_pointer();
 setmin_kernel<<<NN,1>>>(d_t,d_f,d_x,d_z, size);

};
__global__ void setmin_kernel(LongPointer *d_t, LongPointer *d_f,unsigned long long int *d_x,unsigned long long int *d_z, int size)
{
   setmin(d_t,d_f,d_x,d_z,size);
};
__device__ void setmin(LongPointer *d_t, LongPointer *d_f,unsigned long long int *d_x,unsigned long long int *d_z, int size )
{    unsigned long long int *col_t,*col_f,m,b,y,x,z;
     x=d_x[blockIdx.x];
     z=0;
	 for (int i=0;i<size;i++)
	 {
		 col_t=_col(d_t,i);
		 col_f=_col(d_f,i);
		 b=col_t[blockIdx.x];
		 y=col_f[blockIdx.x];
		 m=b^y;
		 m&=x;
		 b=y&(~b);
		 b&=x;
		 z|=b;
		 x&=~m;
	 }
	 d_x[blockIdx.x]=x;
	 d_z[blockIdx.x]=z;
};

void SETMAX(Table *T, Table *F, Slice *X, Slice *Z)
{unsigned long long int *d_x,*d_z;
LongPointer *d_t, *d_f;
int NN=X->NN;
int size=T->size;
d_x=X->get_device_pointer();
d_z=Z->get_device_pointer();
d_t=T->get_device_pointer();
d_f=F->get_device_pointer();
 setmax_kernel<<<NN,1>>>(d_t,d_f,d_x,d_z,size);

};
__global__ void setmax_kernel(LongPointer *d_t, LongPointer *d_f,unsigned long long int *d_x,unsigned long long int *d_z,int size)
{
   setmax(d_t,d_f,d_x,d_z,size);
};
__device__ void setmax(LongPointer *d_t, LongPointer *d_f,unsigned long long int *d_x,unsigned long long int *d_z,int size )
{    unsigned long long int *col_t,*col_f,m,b,y,x,z;
     x=_assign(d_x);
     z=0;
	 for (int i=0;i<size;i++)
	 { col_t=_col(d_t,i);
	 col_f=_col(d_f,i);
	 b=_assign(col_t);
	 y=_assign(col_f);
	 m=b^y;
	 m&=x;
	 b=(~y)&(b);
	 b&=x;
	 z|=b;
	 x&=~m;
 }
 _assign(d_x,x);
 _assign(d_z,z);
};

void HIT(Table *T, Table *F, Slice *X, Slice *Z)
{unsigned long long int *d_x,*d_z;
LongPointer *d_t, *d_f;
int NN=X->NN;
int size=T->size;
d_x=X->get_device_pointer();
d_z=Z->get_device_pointer();
d_t=T->get_device_pointer();
d_f=F->get_device_pointer();
 hit_kernel<<<NN,1>>>(d_t,d_f,d_x,d_z,size);
};
__global__ void hit_kernel(LongPointer *d_t, LongPointer *d_f,unsigned long long int *d_x,unsigned long long int *d_z,int size)
{
   hit(d_t,d_f,d_x,d_z,size);
};
__device__ void hit(LongPointer *d_t, LongPointer *d_f,unsigned long long int *d_x,unsigned long long int *d_z, int size )
{    unsigned long long int *col_t,*col_f,b,y,z;
     z=_assign(d_x);
	 for (int i=0;i<size;i++)
	 {
		 col_t=_col(d_t,i);
		 col_f=_col(d_f,i);
		 b=_assign(col_t);
		 y=_assign(col_f);
		 y^=b;
		 z&=~y;
	 }
	 _assign(d_z,z);
};

void TMERGE(Table *T,  Slice *X, Table *F,int k)
{
	unsigned long long int *d_x;
	LongPointer *d_t, *d_f;
	int NN=X->NN;
	int size=T->size;
	d_x=X->get_device_pointer();
	d_t=T->get_device_pointer();
	d_f=F->get_device_pointer();
	 tmerge_kernel<<<NN,k>>>(d_t,d_x,d_f,size);

};
__global__ void tmerge_kernel(LongPointer *d_t,unsigned long long int *d_x, LongPointer *d_f,int size)//<<<NN,1...M>>>
{   if (gridDim.y==1)
	{
	    tmerge(d_t,d_x,d_f,size);
//	    printf("1D parallization");
	}
    else
    {
    	tmerge_par(d_t,d_x,d_f);
//    	printf("2D parallization\n");
    }
//	printf("<<(),(%i,%i)>>",blockDim.x,blockDim.y);
};
__device__ void tmerge(LongPointer *d_t,unsigned long long int *d_x, LongPointer *d_f,int size)
{   unsigned long long int *col_t,*col_f,y,z,a,x;
     x=_assign(d_x);
     a=~x;
	 for (int i=0;i<size;i++)
//     int i=blockIdx.y;
		 {
		 col_t=_col(d_t,i);
		 col_f=_col(d_f,i);
		 y=_assign(col_t)&x;
		 z=_assign(col_f)&a;
		 z|=y;
		 _assign(col_f,z);
		 }
//		 printf("b_x=%i,b_y=%i,thr_x=%i,thr_y=%i ,i=%i\n",blockIdx.x,blockIdx.y, threadIdx.x,threadIdx.y,i);
};

__device__ void tmerge_par(LongPointer *d_t,unsigned long long int *d_x, LongPointer *d_f)
{   unsigned long long int *col_t,*col_f,y,z,a,x;
     x=_assign(d_x);
     a=~x;
//	 for (int i=0;i<M;i++)
     int i=blockIdx.y;
		 {
		 col_t=_col(d_t,i);
		 col_f=_col(d_f,i);
		 y=_assign(col_t)&x;
		 z=_assign(col_f)&a;
		 z|=y;
		 _assign(col_f,z);
		 }
//		 printf("b_x=%i,b_y=%i,thr_x=%i,thr_y=%i ,i=%i\n",blockIdx.x,blockIdx.y, threadIdx.x,threadIdx.y,i);
};

void WMERGE(Slice *v,  Slice *X, Table *F,int k)
{
	unsigned long long int *d_x,*d_v;
	LongPointer *d_f;
	int NN=X->NN;
	d_v=v->get_device_pointer();
	d_x=X->get_device_pointer();
	d_f=F->get_device_pointer();
    int MM=F->size;
	 wmerge_kernel<<<NN,k>>>(d_v,d_x,d_f,MM);
};
__global__ void wmerge_kernel(unsigned long long int *d_v,unsigned long long int *d_x, LongPointer *d_f,int MM)//<<<NN,1...M>>>
{
	wmerge(d_v,d_x,d_f,size,MM);
};
__device__ void wmerge(unsigned long long int *d_v,unsigned long long int *d_x, LongPointer *d_f, int MM)
{ unsigned long long int *col_f,y,z,x;
  int pos;
   x=_assign(d_x);
   y=~x;
   for (int i=threadIdx.x;i<MM;i=i+blockDim.x)
  		 {
          col_f=_col(d_f,i);
          z=_assign(col_f)&y;
          pos=_get_bit(d_v,i+1);
          if (pos==1)z|=x;
 	   	  _assign(col_f,z);
  		 }
};

void WCOPY(Slice *v,  Slice *X, Table *F,int k)
{
	unsigned long long int *d_x,*d_v;
	LongPointer *d_f;
	int NN=X->NN;
    int size=F->size;
	d_v=v->get_device_pointer();
	d_x=X->get_device_pointer();
	d_f=F->get_device_pointer();
	 wcopy_kernel<<<NN,k>>>(d_v,d_x,d_f,size);
};
__global__ void wcopy_kernel(unsigned long long int *d_v,unsigned long long int *d_x, LongPointer *d_f,int k)//<<<NN,1...M>>>
{
	wcopy(d_v,d_x,d_f,k);
};
__device__ void wcopy(unsigned long long int *d_v,unsigned long long int *d_x, LongPointer *d_f,int k)
{ unsigned long long int *col_f,y;
     int pos;
     y=0;
     for (int i=threadIdx.x;i<k;i=i+blockDim.x)
		 {
    	  col_f=_col(d_f,i);
    	  pos=_get_bit(d_v,i+1);
    	  if (pos==1)
    	  {   col_f[blockIdx.x]=d_x[blockIdx.x];
    		  //_assign(col_f,d_x);
    	  }
    	  else
    	  { // _assign(col_f,y);
    	    col_f[blockIdx.x]=y;
    	  }
		 }
};

void TCOPY(Table *T, Table *F,int k)
{
	LongPointer *d_t, *d_f;
	int NN=(T->length-1)/SIZE_OF_LONG_INT+1;
    int r=T->size;
 //   printf("NN=%i, r=%i", NN, r);
	d_t=T->get_device_pointer();
	d_f=F->get_device_pointer();
	if (k>r)k=r;
	 tcopy_kernel<<<NN,k>>>(d_t,d_f,r);

};
__global__ void tcopy_kernel(LongPointer *d_t, LongPointer *d_f,int r)
{
	tcopy(d_t,d_f,r);
};
__device__ void tcopy(LongPointer *d_t, LongPointer *d_f,int r)
{ unsigned long long int *col_f, *col_t,y;
     for (int i=threadIdx.x;i<r;i=i+blockDim.x)
		 {
    	    col_t=_col(d_t,i);
	        col_f=_col(d_f,i);
//	        y=_assign(col_t);
//	        _assign(col_f,y);
	        _assign(col_f,col_t);
//	        printf("copy %i %llu=%llu \n",i,d_t[i],d_f[i]);
		 }
};

void TCOPY1(Table *T,int j, int h, Table *F,int k)
{
	LongPointer *d_t, *d_f;
	int NN=(T->length-1)/SIZE_OF_LONG_INT+1;

//	 cudaError_t err = cudaGetLastError();
//		 printf("error befor TCOPY1 %d \n",err);

	d_t=T->get_device_pointer();
	d_f=F->get_device_pointer();
	if ((k==1) || (k>h))k=h;
//	printf("=============================TCOPY1=%i<%i,%i>======================\n",j,NN,k);
	 tcopy1_kernel<<<NN,k>>>(d_t,j,h,d_f);
//	  err = cudaGetLastError();
//	 printf("error after TCOPY1 %d \n",err);
};
__global__ void tcopy1_kernel(LongPointer *d_t, int j,int h, LongPointer *d_f)//<<<NN,k>>> k=1,...,h
{
//	printf("TCOPY1 %i %i\n", j, h);
	tcopy1(d_t,j,h,d_f);
};
__device__ void tcopy1(LongPointer *d_t, int j, int h, LongPointer *d_f)
{    unsigned long long int *col_f, *col_t,y;
//printf("tcopy1===============i=%i[%i]   \n", threadIdx.x,blockIdx.x);
     int k=(j-1)*h;
     for (int i=threadIdx.x;i<h;i=i+blockDim.x)
	 {
	    col_t=_col(d_t,k+i);
        col_f=_col(d_f,i);

//        y=col_t[blockIdx.x];
//       col_f[blockIdx.x]=y;
        col_f[blockIdx.x]=col_t[blockIdx.x];

//   if((threadIdx.x==0)||(threadIdx.x==(blockDim.x-1))) printf("strip %i: <%i,%i> i=%i k=%i \n",j,threadIdx.x,blockIdx.x, i,k+i);
	 }
};

void TCOPY2(Table *T,int j, int h, Table *F,int k)
{
	LongPointer *d_t, *d_f;
	int NN=(T->length-1)/SIZE_OF_LONG_INT+1;
	d_t=T->get_device_pointer();
	d_f=F->get_device_pointer();
	if ((k==1) || (k>h))k=h;
//printf(" TCOPY2 NN=%i \t k=%i \t",NN,k);
	 tcopy2_kernel<<<NN,k>>>(d_t,j,h,d_f);
//puts("TCOPY2 done");
};
__global__ void tcopy2_kernel(LongPointer *d_t, int j,int h, LongPointer *d_f)//<<<NN,k>>> k=1,...,h
{
//printf("tcopy2_kernel \n");
	tcopy2(d_t,j,h,d_f);
};
__device__ void tcopy2(LongPointer *d_t, int j, int h, LongPointer *d_f)
{    unsigned long long int *col_f, *col_t,y;
     int k=(j-1)*h+threadIdx.x;
     for (int i=threadIdx.x;i<h;i=i+blockDim.x)
	 {
	    col_t=_col(d_t,i);
        col_f=_col(d_f,k);
 //printf("T \t %i -> F \t %i \n",i,k);
  //     y=_assign(col_t);
  //     _assign(col_f,y);
       _assign(col_f,col_t);
       k=k+blockDim.x;
	 }
	;
};

void ADDV(Table *T, Table *R, Slice *X, Table *S)
{
	LongPointer *d_t, *d_r,*d_s;
	unsigned long long int *d_x, *d_b;
	int h=T->size;
	int NN=X->NN;
	Slice *B;
	B=new Slice;
    B->Init(LENGTH1);

	d_t=T->get_device_pointer();
	d_r=R->get_device_pointer();
	d_s=S->get_device_pointer();
	d_x=X->get_device_pointer();
	d_b=B->get_device_pointer();

	 addv_kernel<<<NN,1>>>(d_t,d_r,h,d_x,d_s,d_b);

//	 if (Y->SOME()) S->With(Y);
};
__global__ void addv_kernel(LongPointer *d_t,LongPointer *d_r,int h,unsigned long long int *d_x,LongPointer *d_s,unsigned long long int *d_b)
{
	addv(d_t,d_r,h,d_x,d_s,d_b);
}

__device__ void addv(LongPointer *d_t,LongPointer *d_r,int h,unsigned long long int *d_x,LongPointer *d_s,unsigned long long int *d_b)
{ unsigned long long int x,b,y,z,m,*col_t, *col_r,*col_s;
  m=0;
  x=_assign(d_x);
  for (int i=h-1;i>=0;i--)
  {
	  col_t=_col(d_t,i);
      y=_assign(col_t)&x;
      col_r=_col(d_r,i);
      z=_assign(col_r)&x;
      b=y&z;
      z^=y;
      y=z^m;
      col_s=_col(d_s,i); _assign(col_s,y);
      y=z & m;
      b|=y;
      m=b;
  }
  _assign(d_b,b);
};


void ADDC(Table *T, Slice *w, Slice *X, Table *S)
{
	LongPointer *d_t,*d_s;
	unsigned long long int *d_x, *d_b,*d_w;
	int h=T->size;
	int NN=X->NN;
	Slice *B;
	B=new Slice;
    B->Init(LENGTH1);

	d_t=T->get_device_pointer();
	d_w=w->get_device_pointer();
	d_s=S->get_device_pointer();
	d_x=X->get_device_pointer();
	d_b=B->get_device_pointer();

	 addc_kernel<<<NN,1>>>(d_t,d_w,h,d_x,d_s,d_b);

	 if (B->SOME()) puts("AddC is incorrect.");// S->With(Y);
};
__global__ void addc_kernel(LongPointer *d_t,unsigned long long int *d_w,int h,unsigned long long int *d_x,LongPointer *d_s,unsigned long long int *d_b)
{
	addc(d_t,d_w,h,d_x,d_s,d_b);
};
__device__ void addc(LongPointer *d_t,unsigned long long int *d_w,int h,unsigned long long int *d_x,LongPointer *d_s,unsigned long long int *d_b)
{ unsigned long long int y,m,*col_t,*col_s,b,x;
 int pos;
  b=0;
 // x=_assign(d_x);
  x=d_x[blockIdx.x];
  for (int i=h-1;i>=0;i--)
  {
	  col_t=_col(d_t,i);
//      y= _assign(col_t);
	  y=col_t[blockIdx.x];
      m=b^y;
      pos=_get_bit(d_w,i+1);
      if (pos==0)
      {	  b&=y;
//      printf("x:%llu \t %llu 0",x,y, m);
      }
      else
      {
    	  m=~m;
    	  b|=y;
//    	printf("x:%llu \t %llu 1",x,y, m);
      }
      col_s=_col(d_s,i);
      m&=x;
      //_assign(col_s,m);
      col_s[blockIdx.x]=m;
//    printf("%llu\n" m);
      b&=x;

   }
 // if (b!=0) printf("addc is incorrect\n");
  _assign(d_b,b);
};

int ADDC1( Slice *X, Slice *w, Table *S)
{	LongPointer *d_s;
unsigned long long int *d_x, *d_b,*d_w;
int h=S->size;
int k,i,j;
int NN=X->NN;
Slice *B;
B=new Slice;
B->Init(LENGTH1);

d_w=w->get_device_pointer();
d_s=S->get_device_pointer();
d_x=X->get_device_pointer();
d_b=B->get_device_pointer();

 addc1_kernel<<<NN,1>>>(d_x,d_w,h,d_s,d_b);

	 if (B->SOME())
	 {	// puts("AddC1 is incorrect.");//S->With(Y) and error;
	 //    B->print("ADDC1_error",0);
//		k=B->FND();
//		 printf("B(%i)=1 \n",k);
	     return k;
	 }
	 else return 0;
};
__global__ void addc1_kernel(unsigned long long int *d_x,unsigned long long int *d_w,int h,LongPointer *d_s,unsigned long long int *d_b)
{
	addc1(d_x,d_w,h,d_s,d_b);
};
__device__ void addc1(unsigned long long int *d_x, unsigned long long int *d_w,int h,LongPointer *d_s,unsigned long long int *d_b)
{ unsigned long long int y,y1,z,m,x,nx,*col_s,b;
 int pos;
   b=0;
   x=d_x[blockIdx.x];
   nx=~x;
   m=0;
   for (int i=h;i>0;i--)
   {
//printf("addc1 i=%i\n",i);
 	  col_s=_col(d_s,i-1);
       y= col_s[blockIdx.x];
       m=b^y;
       pos=_get_bit(d_w,i);
       if (pos==0) b&=y;
       else
       {
     	  m=~m;
     	  b|=y;
       }
       m&=x;
       m|=(y&nx);//
       col_s[blockIdx.x]=m;
       b&=x;
/*/
	   col_s=_col(d_s,i-1);
	   y= col_s[blockIdx.x];
	   y1=y&nx;//неизменяемая часть;
	   y&=x;
	   m=b^y;
	   pos=_get_bit(d_w,i);
	   if (pos==0) z=0;
	   else z=x;
	   b=y&z;
	   z^=y;
	   col_s[blockIdx.x]=(z^m)|y1;
	   b|=y;
	   m=b;
	  // */
    }
//   if (b==0) puts("add1c is incorrect");
   d_b[blockIdx.x]=b&x;
   };

void SUBTV(Table *T, Table *R, Slice *X,Table *S)
{
	LongPointer *d_t, *d_r,*d_s;
		unsigned long long int *d_x, *d_b;
		int h=T->size;
		int NN=X->NN;

		Slice *B;
		B=new Slice;
	    B->Init(LENGTH1);

		d_t=T->get_device_pointer();
		d_r=R->get_device_pointer();
		d_s=S->get_device_pointer();
		d_x=X->get_device_pointer();
		d_b=B->get_device_pointer();
//         printf("length=%i, NN=%i, blocks=%i, threads=%i \n",LENGTH1, NN1, blocks1,threads1);
		 subtv_kernel<<<blocks1,threads1>>>(d_t,d_r,h,d_x,d_s,d_b);
 //        B->print("subtv_error.dat",0);
//		 if (B->SOME()) error;
};
__global__ void subtv_kernel(LongPointer *d_t, LongPointer *d_r,int k, unsigned long long int *d_x, LongPointer *d_s,unsigned long long int *d_m)
{
	subtv(d_t,d_r,k,d_x,d_s,d_m);
};
__device__ void subtv(LongPointer *d_t, LongPointer *d_r,int k, unsigned long long int *d_x, LongPointer *d_s,unsigned long long int *d_m)
{ unsigned long long int *col_t, *col_r, *col_s;
  unsigned long long int a,b,p,y,z,m,x;

  m=0;
  x=_assign(d_x);
//  int ind= index;
//  printf("_%i_",ind);
  for (int i=k-1;i>-1;i--)
  {
	  col_t=_col(d_t,i);
	  col_r=_col(d_r,i);
      col_s=_col(d_s,i);
	  z=_assign(col_t);
	  z&=x;
	  y=_assign(col_r);
	  y&=x;
	  p=y^m;
	  a=(p^z)&x;
	  _assign(col_s,a);
	  b=p&(~z);
	  b|=y&m;
	  m=b;
// if((index>=NN))
//	 printf("i=%i ,ind=%i,m=%llu, a==%llu \n",i, ind,m,a);
  }
  _assign(d_m,m);
};

void SUBTC(Table *T, Slice *X, Slice *w, Table *S)
{	LongPointer *d_t,*d_s;
unsigned long long int *d_x, *d_b, *d_w;
int h=T->size;
int NN=X->NN;
Slice *B;
B=new Slice;
B->Init(LENGTH1);

d_t=T->get_device_pointer();
d_w=w->get_device_pointer();
d_s=S->get_device_pointer();
d_x=X->get_device_pointer();
d_b=B->get_device_pointer();

 subtc_kernel<<<NN,1>>>(d_t,d_x,d_w,d_s,h,d_b);

//	 if (B->SOME()) error;
};
__global__ void subtc_kernel(LongPointer *d_t, unsigned long long int *d_x, unsigned long long int *d_w, LongPointer *d_s,int k,unsigned long long int *d_m)
{
	subtc(d_t,d_x,d_w,d_s,k,d_m);
};
__device__ void subtc(LongPointer *d_t, unsigned long long int *d_x, unsigned long long int *d_w, LongPointer *d_s,int k,unsigned long long int *d_m)
{   unsigned long long int *col_t, *col_s;
    unsigned long long int a,b,t,m,s,x;
    int pos;
    x=_assign(d_x);
    m=0;
    for (int i=k-1;i>-1;i--)
    { col_t=_col(d_t,i);
	  col_s=_col(d_s,i);
	  t=_assign(col_t);
	  a=m^t;
	  pos=_get_bit(d_w,i+1);
	       if (pos==0)
	       {
	    	b=m&(~t);
	    	s=a;
	       }
	       else
	       {
	       s=~a;
	       b=s|m;
	       }
	       s&=x;
	       _assign(col_s,s);
	       b&=x;
	       m=b;

    }
    _assign(d_m,m);
};

void SUBTC1(Table *T, Slice *X, Slice *w, Table *S)
{	LongPointer *d_t,*d_s;
unsigned long long int *d_x, *d_b, *d_w;
int h=T->size;
int NN=X->NN;
Slice *B;
B=new Slice;
B->Init(LENGTH1);

d_t=T->get_device_pointer();
d_w=w->get_device_pointer();
d_s=S->get_device_pointer();
d_x=X->get_device_pointer();
d_b=B->get_device_pointer();

 subtc1_kernel<<<NN,1>>>(d_t,d_x,d_w,d_s,h,d_b);

//	 if (B->SOME()) error;
};
__global__ void subtc1_kernel(LongPointer *d_t, unsigned long long int *d_x, unsigned long long int *d_w, LongPointer *d_s,int k,unsigned long long int *d_m)
{
	subtc1(d_t,d_x,d_w,d_s,k,d_m);
};
__device__ void subtc1(LongPointer *d_t, unsigned long long int *d_x, unsigned long long int *d_w, LongPointer *d_s,int k,unsigned long long int *d_m)
{      unsigned long long int *col_t, *col_s;
unsigned long long int a,b,t,m,s,x;
int pos;
x=_assign(d_x);
m=0;
for (int i=k-1;i>-1;i--)
{ col_t=_col(d_t,i);
  col_s=_col(d_s,i);
  t=_assign(col_t);
  a=m^t;
  pos=_get_bit(d_w,i+1);
       if (pos==0)
       {
    	b=m&(~t);
    	s=a;
       }
       else
       {
       s=~a;
       b=s|m;
       }
       s&=x;
       s|=(~x)&t;
       _assign(col_s,s);
       b&=x;
       m=b;
}
_assign(d_m,m);
};

void CLEAR(Table *T)
{};

__device__ void clear(LongPointer *d_tab, int h)
{   unsigned long long int *col_tab,y=0;

    for (int i=threadIdx.x;i<h;i=i+blockDim.x)
    {
      col_tab=_col(d_tab,i);
     _assign(col_tab,y);
    }
};


void WTRANS(Slice *w, int h, Table *R)
{
  wtrans_kernel<<<NN1,SIZE_OF_LONG_INT>>>(w->get_device_pointer(),h,R->length,R->get_device_pointer());
};

__global__ void wtrans_kernel(unsigned long long int *d_w, int h, int length, LongPointer *d_r)
{
  wtrans(d_w,h,length,d_r);
};

__device__ void wtrans(unsigned long long int *d_w, int h, int length, LongPointer *d_r)
{
	unsigned int bid=blockIdx.x;
	unsigned int tid=threadIdx.x;
	unsigned int M1=(length>SIZE_OF_LONG_INT*(bid+1))?SIZE_OF_LONG_INT:length%SIZE_OF_LONG_INT;
	__shared__ unsigned long long  int tmp[SIZE_OF_LONG_INT];
	unsigned int i,r_i;//, M1=SIZE_OF_LONG_INT;
	unsigned long long int *d_n;
	r_i=h*tid+SIZE_OF_LONG_INT*bid;
	for (i=0; i<h;i++)
	{
		tmp[tid]=0;
		d_n=d_r[i];
		if(get_position_bit(d_w,r_i)==1)
		{
//если в позиции 1
		 tmp[tid]=1<<tid;

		}
		if(tid<M1) printf("col %i,%i,%i, <%i, %i>, pos %i =%i \n",i,length,M1,bid,tid,r_i,get_position_bit(d_w,r_i));
		d_n[bid]=  get_array(tmp,0,M1)
		  	    	    	|  get_array(tmp,1,M1)
		  	    	    	|  get_array(tmp,2,M1)
		  	    	    	|  get_array(tmp,3,M1)
		  	    	        |  get_array(tmp,4,M1)
		  	    	        |  get_array(tmp,5,M1)
		  	    	        |  get_array(tmp,6,M1)
		  			        |  get_array(tmp,7,M1)
		  	         		|  get_array(tmp,8,M1)
		  			        |  get_array(tmp,9,M1)
		  	                |  get_array(tmp,10,M1)
		  	                |  get_array(tmp,11,M1)
		  	                |  get_array(tmp,12,M1)
			                |  get_array(tmp,13,M1)
			                |  get_array(tmp,14,M1)
			                |  get_array(tmp,15,M1)
		                    |  get_array(tmp,16,M1)
		                    |  get_array(tmp,17,M1)
		                    |  get_array(tmp,18,M1)
		          			|  get_array(tmp,19,M1)
					        |  get_array(tmp,20,M1)
				            |  get_array(tmp,21,M1)
		                    |  get_array(tmp,22,M1)
		                    |  get_array(tmp,23,M1)
		                    |  get_array(tmp,24,M1)
		   	                |  get_array(tmp,25,M1)
			                |  get_array(tmp,26,M1)
			                |  get_array(tmp,27,M1)
		                    |  get_array(tmp,28,M1)
		                    |  get_array(tmp,29,M1)
		                    |  get_array(tmp,30,M1)
			                |  get_array(tmp,31,M1)
				            |  get_array(tmp,32,M1)
				            |  get_array(tmp,33,M1)
		                    |  get_array(tmp,34,M1)
		                    |  get_array(tmp,35,M1)
		                    |  get_array(tmp,36,M1)
		   	                |  get_array(tmp,37,M1)
			                |  get_array(tmp,38,M1)
			                |  get_array(tmp,39,M1)
		                    |  get_array(tmp,40,M1)
		                    |  get_array(tmp,41,M1)
		                    |  get_array(tmp,42,M1)
					        |  get_array(tmp,43,M1)
			            	|  get_array(tmp,44,M1)
			             	|  get_array(tmp,45,M1)
		                    |  get_array(tmp,46,M1)
		                    |  get_array(tmp,47,M1)
		                    |  get_array(tmp,48,M1)
		    	            |  get_array(tmp,49,M1)
			                |  get_array(tmp,50,M1)
			                |  get_array(tmp,51,M1)
		                    |  get_array(tmp,52,M1)
		                    |  get_array(tmp,53,M1)
		                    |  get_array(tmp,54,M1)
				            |  get_array(tmp,55,M1)
			         	    |  get_array(tmp,56,M1)
				            |  get_array(tmp,57,M1)
		                    |  get_array(tmp,58,M1)
		                    |  get_array(tmp,59,M1)
		                    |  get_array(tmp,60,M1)
		                    |  get_array(tmp,61,M1)
		                    |  get_array(tmp,62,M1)
		                    |  get_array(tmp,63,M1);

		r_i++;
	}

};

// d_x - строка, по которой вставляют
// k - номер столбца, который вставляют
__global__ void x_w_table_or_kernel(LongPointer *d_tab, unsigned long long int *d_w, unsigned long long int *d_x)
{	 unsigned long long int *d_v;
	        int s;

		//    d_w=d_tab[k-1];//tab->GetCol(w,k);
		    s=_get_bit(d_x,threadIdx.x+1);
	    	//s=get_position_bit(d_x,threadIdx.x+1);
	    	if(s==1)
	    	{
	    		d_v=d_tab[threadIdx.x];//tab->GetCol(w,k);
	    		_or(d_v,d_w);

	    	}
}

__global__ void x_w_table_and_kernel(LongPointer *d_tab, unsigned long long int *d_w, unsigned long long int *d_x)
{	 unsigned long long int *d_v;
     int s;
//    d_w=d_tab[k-1];//tab->GetCol(w,k);
    s=_get_bit(d_x,threadIdx.x+1);
//s=get_position_bit(d_x,threadIdx.x+1);
    if(s==1)
	{
		d_v=d_tab[threadIdx.x];//tab->GetCol(w,k);
	   _and(d_v,d_w);
   	}
}


void TCOPY3(Table *T,int j, int h, Table *F,int k)
{
	LongPointer *d_t, *d_f;
	int NN=(F->length-1)/SIZE_OF_LONG_INT+1;
	d_t=T->get_device_pointer();
	d_f=F->get_device_pointer();
	int h1=F->size;
	if ((k==1) || (k>h1))k=h1;
//printf(" TCOPY2 NN=%i \t k=%i \t",NN,k);
	 tcopy3_kernel<<<NN,k>>>(d_t,j,h,d_f);
//puts("TCOPY2 done");
}

__global__ void tcopy3_kernel(LongPointer *d_t, int j,int h, LongPointer *d_f)//!<<<NN,k>>> k=1,...,f.size
{
	tcopy3(d_t, j,h,d_f);
}

//<<<NN,h>>>
__device__ void tcopy3(LongPointer *d_t, int j, int h, LongPointer *d_f)
{
	unsigned long long int *d_v,*d_v_in;
// проверка, если не все столбцы обрабатываются своей нитью
	d_v=d_f[threadIdx.x];
	d_v_in=d_t[threadIdx.x];
	trim_(d_v,d_v_in,j,h);
}


void TCOPY4(Table *T,int j, int h, Table *F,int k)// Копирует T как горизонтальную полосу в F
{
	LongPointer *d_t, *d_f;
	int NN=(F->length-1)/SIZE_OF_LONG_INT+1;
	d_t=T->get_device_pointer();
	d_f=F->get_device_pointer();
	int h1=F->size;
	if ((k==1) || (k>h1))k=h1;
//printf(" TCOPY2 NN=%i \t k=%i \t",NN,k);
	 tcopy4_kernel<<<NN,k>>>(d_t,j,h,d_f);
//puts("TCOPY4 done");
}
__global__ void tcopy4_kernel(LongPointer *d_t, int j,int h, LongPointer *d_f)//!<<<NN,k>>> k=1,...,h
{
	tcopy4(d_t, j,h,d_f);
}
__device__ void tcopy4(LongPointer *d_t, int j, int h, LongPointer *d_f)
{
	unsigned long long int *d_v,*d_v_in, head, teal;
// проверка, если не все столбцы обрабатываются своей нитью
	d_v=d_f[threadIdx.x];//маленький
	d_v_in=d_t[threadIdx.x];// большой
	int num_el=j>>6;//номер первого элемента в результирующем слайсе
	int num_el1=h>>6; // номер последнего элемента в маленьком
	int num_bit_first= j % SIZE_OF_LONG_INT -1; // номер бита в элементе, в который копируется первый маленького слайса
	int num_bit_last = (j+h-1) % SIZE_OF_LONG_INT;// число элементов
	   char prb[65];
	//   printf("num_els %i and %i (%i) bits from %i to %i \n",num_el,num_el1,num_el2, num_bit_first, num_bit_last);
//	   unsigned long long int teal,head;
	   if (blockIdx.x >num_el)//?????????
	   {      head =d_v_in[blockIdx.x]<<(num_bit_first);
	   	   teal =d_v_in[blockIdx.x-1]>>(SIZE_OF_LONG_INT-num_bit_first);
	   	   d_v[blockIdx.x+num_el]=head | teal;
//	   	   printf("num_els %i (%i)%llu head=%llu (%i)%llu teal=%llu  \n",blockIdx.x,blockIdx.x+num_el,d_v_in[blockIdx.x],head,blockIdx.x-1,d_v_in[blockIdx.x-1],teal);
	   }

}


__device__ void tcopy_hor(LongPointer *d_t, int i,int j, int h, LongPointer *d_f)
// d_t - откуда
// i - первый бит из d_t
// j - первый бит из d_f
// h - сколько бит/строк копируется
// d_f -куда
{
	unsigned long long int *d_v,*d_v_in, head, teal;
// проверка, если не все столбцы обрабатываются своей нитью
	d_v=d_f[threadIdx.x];//маленький
	d_v_in=d_t[threadIdx.x];// большой
	int n1m_el=i>>6;//номер первого элемента в исходном слайсе
	int n2m_el=j>>6;//номер первого элемента в исходном слайсе
	int n1m_el1=(i-1+h)>>6; // номер последнего элемента в маленьком
	int n2m_el1=(j-1+h)>>6; // номер последнего элемента в маленьком
//	int num_bit_first= i % SIZE_OF_LONG_INT -1; // номер бита в элементе, в который копируется первый маленького слайса
//	int num_bit_last = h % SIZE_OF_LONG_INT;
	   char prb[65];
	//   printf("num_els %i and %i (%i) bits from %i to %i \n",num_el,num_el1,num_el2, num_bit_first, num_bit_last);
//	   unsigned long long int head, teal;
     if (i==j) //without shift
     {
    	 if((blockIdx.x>0)&&(blockIdx.x<10))
    	 { d_v_in[blockIdx.x+n2m_el]=d_v[blockIdx.x+n1m_el];}
    	 else {}
     }
     else
    	 if(i<j) //shift_down(j-i)
    	 {

    	 }
    	 else //shift_up(i-j)
    	 {

    	 }
}
