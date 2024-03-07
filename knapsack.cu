#include "knapsack.h"
#include <ctime>
#include <stdio.h>
#include <sys/time.h>
#include "basic.h"
//#include "cuPrintf.cuh"
//#include "cuPrintf.cu"

#define WMAX 100
#define CMAX 1000
#define N_K  16

extern  const int NN1;


// для заданного числа n генерирует масив весов от 1 до WMAX и массив стоимости от 1 до CMAX
void problem_generate(int n, int *w, int *c)
{   srand( time(0));
	int i;
	for (i=0;i<n; i++)
	{
		w[i]=rand()%WMAX+1;
		c[i]=rand()%CMAX+1;
	}
}

// vjue
void branch_cut(int n,int *w, int W, Slice *T,Slice *B)
{   int i,k_tmp=0;
int w_t=W;
int w_b=W;

for (i=0;i<n;i++)
{
	if (w[i]<=w_b)
	{   B->set(i,1);
		w_b-=w[i];

	}
}

for (i=1;i<n+1;i++)
	{
	if(w[n-i]<=w_t)
			{
				w_t-=w[n-i];
				T->set(n-i,1);
			}
}
}

// подряд идущие, без пропусков
void branch_cut(int n, int *w, int W, int &k_t, int &k_b)
{   int i,k_tmp=0;
	int w_t=W;
	int w_b=W;
	k_t=n;
	k_b=0;
	for (i=0;i<n;i++)
	{
		if (w[i]<=w_b)
		{//   printf("%i",1);
			w_b-=w[i];
			k_b++;

		}else
		{
		//	printf("%i",0);
			if (k_tmp==0) k_tmp=k_b;
		}
	}
   // puts("");
	k_b=(k_tmp==0)?k_b:k_tmp+1;
	i=1; k_tmp=0;
//	puts("Mirrored");
	for (i=1;i<n+1;i++)
		{
		if(w[n-i]<=w_t)
				{ //printf("%i",1);
					w_t-=w[n-i];
					k_t--;
				}
		else
				{
					//printf("%i",0);
					if (k_tmp==0) k_tmp=k_t;
				}
	}
		k_t=(k_tmp==0)?k_t:k_tmp;
	//	printf("   %i:%i %i ",k_b,k_t ,w_t);
	//	 puts("");

}

__global__ void init_stable(LongPointer *d_T,int NN1)
{	unsigned long long int	*d_t,
	init_x[]={0xAAAAAAAAAAAAAAAA,0xCCCCCCCCCCCCCCCC,
		   0xF0F0F0F0F0F0F0F0,0xFF00FF00FF00FF00,
		   0xFFFF0000FFFF0000,0xFFFFFFFF00000000,
		   0xFFFFFFFFFFFFFFFF,0};
    unsigned long long int i,j=1,k;
    i=blockIdx.x;
    		{   d_t=d_T[M-i-1];
    			if (i<6)
    			{
    				for (k=0;k<NN1;k++) d_t[k]=init_x[i];

    			}
    			else
    			if (i<70)//там сдвиг уже не сработает, но длины таблицы <2:70
    			{
    				j=j<<(i-6);
    				for (k=0;k<NN1;k++)
    				   d_t[k]=((k&j)==j)?init_x[6]:init_x[7];
    			}
    		}
}
void initial_search_table(Table *T)
{   printf(" NN=%i sz=%i \n",NN1,M);

	//cudaPrintfInit ();
cudaEvent_t start, stop;
float elapsedTime1;
	     cudaEventCreate(&start);
	     cudaEventCreate(&stop);
	     cudaEventRecord(start, 0);
	init_stable<<<M,1>>>(T->get_device_pointer(),NN1);
	 cudaEventRecord(stop, 0);
	 cudaEventSynchronize(stop);
     cudaEventElapsedTime(&elapsedTime1, start, stop);// in 0.001 sec

     printf("associative time init %f (%i)\n", elapsedTime1, NN1);
LongPointer *d_T=T->get_device_pointer();
	unsigned long long int i,k,j=1;
	unsigned long long int	*d_t;
//	cudaMemcpy(hostT,d_T,sizeof(unsigned int int)*NN1*M,cudaMemcpyDeviceToHost);

	unsigned  long long int tmp,hostT[M][NN1],

	 init_x[]={0xAAAAAAAAAAAAAAAA,0xCCCCCCCCCCCCCCCC,
			   0xF0F0F0F0F0F0F0F0,0xFF00FF00FF00FF00,
			   0xFFFF0000FFFF0000,0xFFFFFFFF00000000,
			   0xFFFFFFFFFFFFFFFF,0};
	double tt;
    struct timeval tv1,tv2;
    gettimeofday(&tv1,NULL);

	for(i=0;i<M;i++)
	{
//		d_t=(T->col(M-i))->get_device_pointer();
		if (i<6)
		{
			//cudaMemset(d_t,(int)init_x[i],NN1*2*sizeof(unsigned int));
//			cudaMemset(d_t,init_x[i],NN1*sizeof(unsigned long long int));
//			cudaMemcpy(hostT,d_t,sizeof(unsigned int int)*NN1*M,cudaMemcpyDeviceToHost);
			for (k=0;k<NN1;k++) hostT[M-1-i][k]=init_x[i];

		}
		else
		{  // j=1;
/*			if(i%2==0)
				cudaMemset(d_t,init_x[6],NN1*sizeof(unsigned long long int));
		    else
		    	cudaMemset(d_t,init_x[5],NN1*sizeof(unsigned long long int));
//			printf("%d: %lp %p\n", i, init_x[6], d_t);
 *
 */
//			printf("\n ==== %d ",i);
			for (k=0;k<NN1;k++)
			{
				hostT[M-1-i][k]=((k&j)==j)?	init_x[6]:init_x[7];
//				printf("%p ",hostT[i][k]);
			}
			j=j<<1;
		}

//		printf("%d: %p %p\n", i, init_x[i%8],hostT[i][0]);
	}
	 gettimeofday(&tv2,NULL);
	 tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
	 printf("time of work seq %f sec \n", tt);
	//cudaPrintfDisplay (stdout, true);
	//cudaPrintfEnd ();

	cudaError_t err = cudaGetLastError();
	printf("after init search table %d , %s \n",err,cudaGetErrorString(err));
}

void knapsack_exp()
{
	int w[M],c[M],W,k_b,k_t,i,j,W_t=0,W_b=0;
	problem_generate(M, w, c);
	W=rand()%(WMAX*M/2)+1;
	Slice *ST,*SB,*X,*w_max;

	ST=new Slice;
	ST->Init(M);
	SB=new Slice;
	SB->Init(M);
	X=new Slice;
	X->Init(LENGTH1);
	w_max=new Slice;
	w_max->Init(N_K);
	w_max->FromDigit(W);

	branch_cut(M, w,W,k_t,k_b);
	branch_cut(M, w,W, ST,SB);
	ST->print("st",0);
	SB->print("sb",0);

	unsigned long long int tmp,dig;
// print results branch_cut
	for (i=0; i<M;i++){
//	  tmp=c[i]<<(64-32);
//	  dig=__brevll(tmp); //переворот в правильную сторону для суммирования
		printf("<%i;%i> \n ",w[i],c[i]);
	}
	printf("\n W=%i \n",W);

/*	for (i=0; i<M;i++)
		if (i<k_t) printf("0");
		else {printf("1");
		W_t+=w[i];
		}
	printf("\n W=%i \n",W_t);

	for (i=0; i<M;i++)
		if (i<k_b) {
			printf("1");
			W_b+=w[i];
		}
		else printf("0");

	printf("\n W=%i \n",W_b);
*/
	Table *T;
	unsigned long long int hostT[M][NN1];
	T =new Table;
	T->Init(LENGTH1,M);
	puts("Init T");

	// инициализация таблицы перебора
	// появление в ней sb - условие окончания перебора
	initial_search_table(T);
//	T->writeToFile("log/init");


	/*
	Var T, WT, CT: Table;
	    Y,Z,Z1,Z2,Z3: Slice;
	    SN,ST,SB: Word(n)
	    v,u: Word(h);
	    */

/////////////////////////////////////////////////
	double tt;
    struct timeval tv1,tv2;
    gettimeofday(&tv1,NULL);
    char str[15];
	Table *WT, *CT;

	WT =new Table;
	WT->Init(LENGTH1,N_K);

	CT =new Table;
	CT->Init(LENGTH1,N_K);

	Slice  *Y,*Z,*Z1,*Z2,*Z3, *SN,*u,*v;

	Y = new Slice;
	Y->Init(LENGTH1);

	Z = new Slice;
	Z->Init(LENGTH1);

	Z1 = new Slice;
	Z1->Init(LENGTH1);

	Z2 = new Slice;
	Z2->Init(LENGTH1);

	Z3 = new Slice;
	Z3->Init(LENGTH1);

	SN = new Slice;
	SN->Init(M);

	v = new Slice;
	v->Init(N_K);

	u = new Slice;
	u->Init(N_K);

    T->GetRow(SN,LENGTH1);
    Z1->CLR();
    Z2->SET();Z2->set(1,0);
    Z3->SET();
    ADDC1(Z3,ST,T);
/*
    ST->print("log/ST",1);
    SN->print("log/SN",1);
    SB->print("log/SB",1);
*/
    printf("Начинаем перебор \n");
    j=0;
/*
 *
   sprintf(str, "log/TST%05d",i);
    T->writeToFile(str);
  */

    while (Z1->ZERO())
    {
/*
    	CLEAR(WT);
    	CLEAR(CT);
       puts("Clear WT and CT");
       for(i=1;i<M;i++)
    	{
          v->FromDigit(w[i+1]);
          u->FromDigit(c[i+1]);
          T->GetCol(Y,i);
          ADDC1(Y,v,WT);
          ADDC1(Y,u,CT);
    	}
       puts("summa done");
       LESS(WT,Z3,w_max,Z);
       puts("Less");
  //     MAX(CT,Z,X);
       puts("MAX");
       if (X->get(1)==0)// смена максимума
       {
          i=X->FND();
          T->GetRow(ST,i);
          T->SetRow(ST,1);
  //        row(1,CT):=row(i,CT);//необязательно, посчитается на следующем шаге
       }
       puts("max changes");
       MATCH(T,Z2,SB,Z1);
       puts("MATCH");
*/
       ADDC1(Z2,SN,T); // к первой строке не добавляется, там максимум
       puts("next step");
       j++;
       printf("%d ",j);
/*       sprintf(str, "log/TST%05d",i);
          T->writeToFile(str);
       T->GetRow(ST,LENGTH1);
       ST->print("log/tt",1);
       */
    }

     cudaError_t err = cudaGetLastError();
     printf("after init search table %d , %s \n",err,cudaGetErrorString(err));

     gettimeofday(&tv2,NULL);
	 tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
	 printf("time of all work seq %f sec \n", tt);

	sprintf(str, "res/NP/test%d.txt",j);
	T->writeToFile("res/NP/test.txt");
}
