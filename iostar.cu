#include "param.h"
#include "find.h"
#include "table.h"
#include "iostar.h"
#include <stdio.h>
//#include <stdlib.h>
#include <sys/time.h>

extern char **tb;//[LENGTH1][M]![LENGTH1][LENGTH1+1];  //table transposed

int **tt;//[size_tt][LENGTH1]
int size_tt;
//VER for matrix of weight or adjacency
//2 for unweighed list of arcs
//3 for weighed list of arcs

void readfromDimageA(char *fn, Table *T )
// read matrix of adjacency from dimage format ('a' l r w)
{	int ras=max(100,M+1);
    char str[ras];

	FILE *f;
	int   n,i,k,l;
	int M2=T->size;
//	int eds=0;
	int length=T->length;

//	puts(" begin set by 0\n");
	for (l=0; l<M2;l++)
	{
		for(int i = 0;i < length;i++)
		{
			tb[i][l] ='0';
		}
	}

	if((f = fopen(fn,"rt")) == NULL)
	{
		puts("file error");
		return;
	}

	while(fgets(str,ras,f) != NULL)
{

  if(str[0]=='a')
  {
	n=atoi(str+1);
	k=n;
	l=1;
	while (k>0)
	{
		k=k/10;
		l++;
	}
	n--;
	i=atoi(str+l+2);
	i--;
	if ((i<M2) and (n<M2)){
		 tb[n][i]='1';

//		 puts(str);
//		 printf("<%i,%i> %c\n",n,i,str[l+3]);
//		 eds++;
	}
  }
  for(int i = 0;i < length;i++)
  {
  	tb[i][M2] = 0;
  }
}
fclose(f);

Slice *s;
s=new Slice;
s->Init(M2);
for(int i = 0;i < M2;i++)
{
//		printf("copying %d slice ",i);
for(int j = 0;j < length;j++)
	{
		str[j] = tb[i][j];
//			printf("%c",tb[j][i]);
	}
//	   	str[M] = 0;
//      printf("\n");

//		printf("slice %d %s \n",i,str);
	s = T->col(i+1);
	*s = str;
}
//    puts("end read from file\n");
}

__global__ void vect_to_strip(int j, int *d_tmp, LongPointer *d_tab)
{
__shared__ int ind_t;
           ind_t=j*H1+blockIdx.x; //номер обрабатываемого столбца полосы
//blockIdx.x - номер столбца в полосе, (H1-blockIdx.x) - позиция бита в слове вектора
//k=threadId.x - номер ull в полосе обрабатывает (k*64...(k+1)*64-1) элемента массива, каждая нить свой столбец.
// в последнем блоке нужно проверить выход за границы вектора

int pos=1<<(H1-blockIdx.x-1);
int ind0=threadIdx.x*SIZE_OF_LONG_INT;
int k=((ind0+SIZE_OF_LONG_INT)>LENGTH1)?(LENGTH1%SIZE_OF_LONG_INT):SIZE_OF_LONG_INT;// параметр цикла or (and)
unsigned long long int number=0;
unsigned long long int pos1=1;
	for (int i=0; i<k;i++)
	{
		if(d_tmp[ind0+i]&pos)
			number|=pos1;
		pos1=pos1<<1;
	}
	// в number собран threadIdx.x-й элемент ind_t-го столбца

	__shared__ LongPointer d_col;
	d_col=d_tab[ind_t];
	d_col[threadIdx.x]=number;
//    if ((blockIdx.x==0)||(blockIdx.x==(gridDim.x-1)))printf("<%i,%i> ",ind_t,threadIdx.x);
}

void readfromDimageW(char *fn, Table *T)
// read matrix of weights from dimage format ('a' l r w)
{
	int ras=100;
	    char str[ras];

		FILE *f;
		int   n,i,j,k,l;
		int M2=size_tt;
	//	int eds=0;
		int length=T->length;
		int *d_tt;

		puts(" begin set by 0\n");
		for (l=0; l<M2;l++)
		{
			for(int i = 0;i < length;i++)
			{
				tt[l][i]=INFINITE;
			}
		}
		puts(" end set by 0\n");
		if((f = fopen(fn,"rt")) == NULL)
		{
			puts("file error");
			return;
		}

		while(fgets(str,ras,f) != NULL)
	{

	  if(str[0]=='a')
	  {
		n=atoi(str+1);
		k=n;
		l=1;
		while (k>0)
		{
			k=k/10;
			l++;
		}
		n--;

		i=atoi(str+l+2);
		k=i;
				while (k>0)
				{
					k=k/10;
					l++;
				}
		i--;
		j=atoi(str+l+3);
//		 printf("<%i,%i,%i> \n",n+1,i+1,j);
		if ((i<VER) && (n<VER))
		{
			d_tt=tt[n];
			 d_tt[i]=j;

	//		 puts(str);
	//		 printf("<%i,%i,%i> \n",n+1,i+1,j);
	//		 eds++;
		}
//		else printf("=======<%i,%i,%i> \n",n+1,i+1,j);
	  }
	}
	 fclose(f);
    puts("file was read");
// tt хранит десятичную матрицу весов на CPU, ее нужно преобразовать в бинарную на GPU.
//    действовать по столбцам
    int *d_tmp;
    LongPointer *d_tab;
    d_tab=T->get_device_pointer();

    cudaError_t err = cudaMalloc(&d_tmp,sizeof(int)*LENGTH1);

    for (j=0; j<M2; j++)
    {
    	cudaMemcpy(d_tmp,tt[j],sizeof(int)*LENGTH1,cudaMemcpyHostToDevice);
 //   	printf("\n column %i\n",j);

    	vect_to_strip<<<H1,NN1>>>(j,d_tmp,d_tab);
    }
    err = cudaGetLastError();
    if (err!=0) printf("after vect_to_strip %d , %s \n",err,cudaGetErrorString(err));
}

void readfromDimageC(char *fn, Table *T)
// read matrix of weights from dimage format ('a' l r w)
{
	int ras=100;
	    char str[ras];

		FILE *f;
		int   n,i,j,k,l;
		int M2=size_tt;
	//	int eds=0;
		int length=T->length;
		int *d_tt;

		puts(" begin set by 0\n");
		for (l=0; l<M2;l++)
		{
			for(int i = 0;i < length;i++)
			{
				tt[l][i]=INFINITE;
			}
		}
		puts(" end set by 0\n");
		if((f = fopen(fn,"rt")) == NULL)
		{
			puts("file error");
			return;
		}

		while(fgets(str,ras,f) != NULL)
	{

	  if(str[0]=='a')
	  {
		n=atoi(str+1);
		k=n;
		l=1;
		while (k>0)
		{
			k=k/10;
			l++;
		}
		n--;

		i=atoi(str+l+2);
		k=i;
				while (k>0)
				{
					k=k/10;
					l++;
				}
		i--;
		j=atoi(str+l+3);
//		 printf("<%i,%i,%i> \n",n+1,i+1,j);
		if ((i<VER) && (n<VER))
		{
			d_tt=tt[i];
			 d_tt[n]=j;

	//		 puts(str);
	//		 printf("<%i,%i,%i> \n",n+1,i+1,j);
	//		 eds++;
		}
//		else printf("=======<%i,%i,%i> \n",n+1,i+1,j);
	  }
	}
	 fclose(f);
    puts("file was read");
// tt хранит десятичную матрицу весов на CPU, ее нужно преобразовать в бинарную на GPU.
//    действовать по столбцам
    int *d_tmp;
    LongPointer *d_tab;
    d_tab=T->get_device_pointer();

    cudaError_t err = cudaMalloc(&d_tmp,sizeof(int)*LENGTH1);

    for (j=0; j<M2; j++)
    {
    	cudaMemcpy(d_tmp,tt[j],sizeof(int)*LENGTH1,cudaMemcpyHostToDevice);
 //   	printf("\n column %i\n",j);

    	vect_to_strip<<<H1,NN1>>>(j,d_tmp,d_tab);
    }
    err = cudaGetLastError();
    if (err!=0) printf("after vect_to_strip %d , %s \n",err,cudaGetErrorString(err));
}

void readfromDimageL(char *fn, Table *L, Table *R)
// read list of unweighed arcs from dimage format ('a' l r w)
{
	int ras=100;
	    char str[ras];

		FILE *f;
		int   n,i,j,k,l;
		int M2=size_tt;
		int eds=0;
		int length=L->length;
		int *tt_left;
		int *tt_right;
         tt_left=tt[0];
		 tt_right=tt[1];
		 printf("tt->%p, tt_left->%p tt_right->%p \n", tt,tt_left,tt_right);
		puts(" begin set by 0\n");
			for(int i = 0;i < length;i++)
			{
				tt_left[i]=0;
				tt_right[i]=0;
			}
		puts(" ended set by 0\n");
		if((f = fopen(fn,"rt")) == NULL)
		{
			puts("file error");
			return;
		}

		while((fgets(str,ras,f) != NULL)&&(eds<LENGTH1))
	{

	  if(str[0]=='a')
	  {
		n=atoi(str+1);
		k=n;
		l=1;
		while (k>0)
		{
			k=k/10;
			l++;
		}
//		n;

		i=atoi(str+l+2);
		k=i;
				while (k>0)
				{
					k=k/10;
					l++;
				}
//		i;
//		j=atoi(str+l+3);
//	 printf("<%i,%i> \n",n,i);
		if ((i<VER) and (n<VER)){
			 tt_left[eds]=n;
			 tt_right[eds]=i;
	//		 puts(str);
//			 printf("<%i,%i> \n",n,i);
			 eds++;
		}
	  }
	}
	 fclose(f);
    puts("file was read");

/*    if((f = fopen("tt.dat","wt")) == NULL)return;
    for(i=0; i<eds;i++)
    {
    	fprintf(f,"%i %i\n",tt_left[i],tt_right[i]);
    }
    fclose(f);*/
// tt хранит десятичный список вершин на CPU, ее нужно преобразовать в бинарную на GPU.
//    действовать по столбцам
    int *d_tmp;
    LongPointer *d_tab;

    j=0;
    cudaError_t err = cudaMalloc(&d_tmp,sizeof(int)*LENGTH1);
//Left
    cudaMemcpy(d_tmp,tt[0],sizeof(int)*LENGTH1,cudaMemcpyHostToDevice);
    puts("vect was copied");
    d_tab=L->get_device_pointer();
    vect_to_strip<<<H1,NN1>>>(j,d_tmp,d_tab);
//Right
    cudaMemcpy(d_tmp,tt[1],sizeof(int)*LENGTH1,cudaMemcpyHostToDevice);
    d_tab=R->get_device_pointer();
    vect_to_strip<<<H1,NN1>>>(j,d_tmp,d_tab);
}
void readfromDimageL(char *fn, Table *L, Table *R, Table *W)
// read list of weighted arcs from dimage format ('a' l r w)
{
	int ras=100;
	    char str[ras];

		FILE *f;
		int   n,i,j,k,l;
		int M2=size_tt;
		int eds=0;
		int length=L->length;
		int *tt_left;
		int *tt_right;
		int *tt_weight;
         tt_left=tt[0];
		 tt_right=tt[1];
		 tt_weight=tt[2];

	//	puts(" begin set by 0\n");
		for(int i = 0;i < length;i++)
					{
						tt_left[i]=0;
						tt_right[i]=0;
					}

		if((f = fopen(fn,"rt")) == NULL)
		{
			puts("file error");
			return;
		}

		while((fgets(str,ras,f) != NULL)&&(eds<LENGTH1))
	{

	  if(str[0]=='a')
	  {
		n=atoi(str+1);
		k=n;
		l=1;
		while (k>0)
		{
			k=k/10;
			l++;
		}

		i=atoi(str+l+2);
		k=i;
				while (k>0)
				{
					k=k/10;
					l++;
				}
		j=atoi(str+l+3);
		if ((i<VER) and (n<VER)){
			 tt_left[eds]=n;
			 tt_right[eds]=i;
			 tt_weight[eds]=j;
	//		 puts(str);
//			 printf("<%i,%i> \n",n,i);
			 eds++;
		}
	  }
	}
	 fclose(f);
    puts("file was red");
// tt хранит десятичную матрицу весов на CPU, ее нужно преобразовать в бинарную на GPU.
//    действовать по столбцам
    int *d_tmp;
        LongPointer *d_tab;


        cudaError_t err = cudaMalloc(&d_tmp,sizeof(int)*LENGTH1);
        j=0;
//Left
    cudaMemcpy(d_tmp,tt[0],sizeof(int)*LENGTH1,cudaMemcpyHostToDevice);
    d_tab=L->get_device_pointer();
    vect_to_strip<<<H1,NN1>>>(j,d_tmp,d_tab);
//Right
    cudaMemcpy(d_tmp,tt[1],sizeof(int)*LENGTH1,cudaMemcpyHostToDevice);
    d_tab=R->get_device_pointer();
    vect_to_strip<<<H1,NN1>>>(j,d_tmp,d_tab);
//Weight
    cudaMemcpy(d_tmp,tt[2],sizeof(int)*LENGTH1,cudaMemcpyHostToDevice);
    d_tab=W->get_device_pointer();
    vect_to_strip<<<H1,NN1>>>(j,d_tmp,d_tab);
}

__global__ void strip_to_vect(int j, int *d_tmp, LongPointer *d_tab)
{
__shared__ int ind_t;
           ind_t=j*H1+threadIdx.x; //номер обрабатываемого столбца полосы
//blockIdx.x - номер столбца в полосе, (H1-blockIdx.x) - позиция бита в слове вектора
//k=threadId.x - номер ull в полосе обрабатывает (k*64...(k+1)*64-1) элемента массива, каждая нить свой столбец.
// в последнем блоке нужно проверить выход за границы вектора
//printf("===================================__global__ strip_to_vect\n============================\n");
int pos=1<<(H1-threadIdx.x-1);
int ind0=blockIdx.x*SIZE_OF_LONG_INT;//number of elements
int k=(((blockIdx.x+1)*SIZE_OF_LONG_INT)>LENGTH1)?(LENGTH1%SIZE_OF_LONG_INT):SIZE_OF_LONG_INT;// параметр цикла or (and)
__shared__ int number;
   number=0;
  LongPointer d_col;
    d_col=d_tab[ind_t];
unsigned long long int num=d_col[blockIdx.x];
//printf("block=%i thread=%i pos=%i k=%i \n",blockIdx.x, threadIdx.x, pos,k);
unsigned long long int pos1=1;
	for (int i=0; i<k;i++)
	{
		if(num&pos1)
		{
			atomicOr(&number,pos);
//			printf("%i %i\n",ind0+k,threadIdx.x);
		}
		__syncthreads();
		pos1=pos1<<1;

		if (threadIdx.x==0)
		{
//			printf("block=%i thread=%i =====tmp[%i]=%i\n", ind0+i, number);
			d_tmp[ind0+i]=number;
			number=0;
		}

		__syncthreads();
	}
	// в number собран threadIdx.x-й элемент ind_t-го столбца

}
__global__ void strip_to_vect1(int j, int *d_tmp, LongPointer *d_tab)
{
__shared__ int ind_t;
           ind_t=(j+1)*H1-1; //номер последнего столбца обрабатываемой  полосы
//blockIdx.x - номер элемента в столбце, (H1-blockIdx.x) - позиция бита в слове вектора
//k=threadId.x - номер строки в элементе.
// в последнем блоке нужно проверить выход за границы вектора

int pos=1;
int ind0=blockIdx.x*SIZE_OF_LONG_INT+threadIdx.x;//number of row
//printf("====ind=%i \n",ind0);
   if(ind0<LENGTH1)
   {
	   int number=0;
	   LongPointer d_col;
	   unsigned long long int num;
	   unsigned long long int pos1=1;
	   	   	   	   pos1<<=threadIdx.x;
	   for(int i=0; i<H1;i++)
	   {
		   d_col=d_tab[ind_t];
		   num=d_col[blockIdx.x];
		   if(num&pos1)
		   {
			   number|=pos;
//			   printf("row=%i i=%i col_numb=%i pos=%i ===============",ind0,i,ind_t,pos);
		   }
		   pos<<=1;
		   ind_t--;
	   }
	   d_tmp[ind0]=number;
//	   printf("====ind=%i th=%i  number=%i\n",ind0,threadIdx.x,number);
   }
}
void writeStrip(char *fn, LongPointer *d_tab, int i)
{   FILE *f;
	int *d_tmp, d_t[LENGTH1];
	int inf=(1<<H1)-1;
	cudaError_t err = cudaMalloc(&d_tmp,sizeof(int)*LENGTH1);
//	 struct timeval tv1,tv2;
//	 double tt;
//	gettimeofday(&tv1,NULL);
	strip_to_vect<<<N1,H1>>>(i,d_tmp,d_tab);
//	gettimeofday(&tv2,NULL);
//			 tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
//			 printf("time of work strip_to_vect \t \t %f sec \n", tt);

//	gettimeofday(&tv1,NULL);
//	strip_to_vect1<<<N1,SIZE_OF_LONG_INT>>>(i,d_tmp,d_tab);
//	gettimeofday(&tv2,NULL);
//				 tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
//				 printf("time of work strip_to_vect1 \t \t %f sec \n", tt);

	cudaMemcpy(d_t,d_tmp,sizeof(int)*LENGTH1,cudaMemcpyDeviceToHost);
//	puts("bin to dec ended");
	if((f = fopen(fn,"wt")) == NULL)return;
	for(int i = 0;i<LENGTH1;i++)
			{
		        if(d_t[i]!=inf)fprintf(f,"%i \n",d_t[i]);
					else fprintf(f,"inf \n");
			}
//	puts("print all");
	fclose(f);
}
void printStrip( LongPointer *d_tab, int i)
{
	int *d_tmp, d_t[LENGTH1];
	int inf=(1<<H1)-1;
	cudaError_t err = cudaMalloc(&d_tmp,sizeof(int)*LENGTH1);
//	 struct timeval tv1,tv2;
//	 double tt;
//	gettimeofday(&tv1,NULL);
	strip_to_vect<<<N1,H1>>>(i,d_tmp,d_tab);
//	gettimeofday(&tv2,NULL);
//			 tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
//			 printf("time of work strip_to_vect \t \t %f sec \n", tt);

//	gettimeofday(&tv1,NULL);
//	strip_to_vect1<<<N1,SIZE_OF_LONG_INT>>>(i,d_tmp,d_tab);
//	gettimeofday(&tv2,NULL);
//				 tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
//				 printf("time of work strip_to_vect1 \t \t %f sec \n", tt);

	cudaMemcpy(d_t,d_tmp,sizeof(int)*LENGTH1,cudaMemcpyDeviceToHost);
//	puts("bin to dec ended");
//	printf("inf=%i \n", INFINITE);
	for(int i = 0;i<LENGTH1;i++)
			{
				if(d_t[i]!=inf)printf("%i ",d_t[i]);
						else printf("inf ");
			}
	printf("\n");
//	puts("print all");
}
void writetoDimageA(char *fn, Table *T )
// write matrix of adjacency to dimage format ('a' l r w)
{ FILE *f;
   int k,j=0;
   Slice *X;
   X=new Slice;
   X->Init(LENGTH1);
//   puts("WtDA 1");
  if((f = fopen(fn,"wt")) == NULL){puts("can not open file");return;}
  for(int i=1;i<=VER;i++)
  {
	  T->GetCol(X,i);
//	  X->print("X",1);
//	  printf("%i \t",i);
//	  puts("WtDA 2");
	  k=X->STEP();
	  while(k>0)
	  {   j++;
		  fprintf(f,"a %i %i\n",i,k);
		  k=X->STEP();
	  }
  }
  puts("WtDA 3");
  fprintf(f,"p sp %i %i\n",VER,j);
  fclose(f);
}

void writetoDimageW(char *fn, Table *T)
// write matrix of weights to dimage format ('a' l r w)
{
	FILE *f;
	int *d_tmp, d_t[LENGTH1];
	cudaError_t err = cudaMalloc(&d_tmp,sizeof(int)*LENGTH1);
	int * d_tt;
	LongPointer *d_tab;
	d_tab=T->get_device_pointer();
 printf("size_tt=%i\n",size_tt);
	for (int i=0; i<size_tt;i++)
	{
		d_tt=tt[i];
		strip_to_vect<<<N1,H1>>>(i,d_tmp,d_tab);
		cudaMemcpy(d_tt,d_tmp,sizeof(int)*LENGTH1,cudaMemcpyDeviceToHost);
	}

	if((f = fopen(fn,"wt")) == NULL)return;
	fprintf(f,"c \np  sp %i %i \nc graph containts %i nodes and %i arcs \n",VER, LENGTH1,VER, LENGTH1);
	for(int i = 0;i<VER;i++)
	{
		for(int j=0;j<LENGTH1;j++)
		{
			if (tt[j][i]!=INFINITE)
			{
				fprintf(f,"a %i %i %i\n",i+1,j+1,tt[j][i]);
			    printf(" %i\t", tt[j][i]);
			}
			else  printf(" --\t");
		}
	printf("\n");
	}
//	puts("print all");
fclose(f);

}

void writetoDimageL(char *fn, Table *L, Table *R)
// write list of unweighed arcs to dimage format ('a' l r w)
{
	  FILE *f;
		int *d_tmp, d_t[LENGTH1];
		int *d_left;
		d_left=tt[0];
		int *d_right;
		int i=0;
		d_right=tt[1];
		cudaError_t err = cudaMalloc(&d_tmp,sizeof(int)*LENGTH1);
/*		 struct timeval tv1,tv2;
		 double tt;
		gettimeofday(&tv1,NULL);*/
		LongPointer *d_tab;
		d_tab=L->get_device_pointer();
		strip_to_vect<<<N1,H1>>>(i,d_tmp,d_tab);
		cudaMemcpy(d_left,d_tmp,sizeof(int)*LENGTH1,cudaMemcpyDeviceToHost);

		d_tab=R->get_device_pointer();
		strip_to_vect<<<N1,SIZE_OF_LONG_INT>>>(i,d_tmp,d_tab);
		cudaMemcpy(d_right,d_tmp,sizeof(int)*LENGTH1,cudaMemcpyDeviceToHost);
	//	puts("bin to dec ended");
		if((f = fopen(fn,"wt")) == NULL)return;
		fprintf(f,"c \np  sp %i %i \nc graph containts %i nodes and %i arcs \n",VER, LENGTH1,VER, LENGTH1);
		for(i = 0;i<LENGTH1;i++)
				{
					fprintf(f,"a %i %i\n",d_left[i],d_right[i]);
				}
	//	puts("print all");
		fclose(f);
}

void writetoDimageL(char *fn, Table *L, Table *R, Table *W)
// write list of weighted arcs to dimage format ('a' l r w)
{
	  FILE *f;
		int *d_tmp, d_t[LENGTH1];
		int *d_left;
		d_left=tt[0];
		int *d_right;
		int i=0;
		d_right=tt[1];
		int *d_weight;
		d_weight=tt[2];
		cudaError_t err = cudaMalloc(&d_tmp,sizeof(int)*LENGTH1);
/*		 struct timeval tv1,tv2;
		 double tt;
		gettimeofday(&tv1,NULL);*/
		LongPointer *d_tab;
		d_tab=L->get_device_pointer();
		strip_to_vect<<<N1,H1>>>(i,d_tmp,d_tab);
		cudaMemcpy(d_left,d_tmp,sizeof(int)*LENGTH1,cudaMemcpyDeviceToHost);

		d_tab=R->get_device_pointer();
		strip_to_vect<<<N1,SIZE_OF_LONG_INT>>>(i,d_tmp,d_tab);
		cudaMemcpy(d_right,d_tmp,sizeof(int)*LENGTH1,cudaMemcpyDeviceToHost);

		d_tab=W->get_device_pointer();
		strip_to_vect<<<N1,SIZE_OF_LONG_INT>>>(i,d_tmp,d_tab);
		cudaMemcpy(d_weight,d_tmp,sizeof(int)*LENGTH1,cudaMemcpyDeviceToHost);

	//	puts("bin to dec ended");
		if((f = fopen(fn,"wt")) == NULL)return;
		fprintf(f,"c \np  sp %i %i \nc graph containts %i nodes and %i arcs \n",VER, LENGTH1,VER, LENGTH1);
		for(i = 0;i<LENGTH1;i++)
				{
					fprintf(f,"a %i %i %i\n",d_left[i],d_right[i],d_weight[i]);
				}
	//	puts("print all");
		fclose(f);
}

void readfromDimageWC(char *fn, Table *T, Table *Cost)
{
	int ras=100;
	    char str[ras];

		FILE *f;
		int   n,i,j,k,l;
		int M2=size_tt;
	//	int eds=0;
		int length=T->length;
		int *d_tt;

		puts(" begin set by 0\n");
		for (l=0; l<M2;l++)
		{
			for(int i = 0;i < length;i++)
			{
				tt[l][i]=INFINITE;
			}
		}
		puts(" end set by 0\n");
		if((f = fopen(fn,"rt")) == NULL)
		{
			puts("file error");
			return;
		}

		while(fgets(str,ras,f) != NULL)
	{

	  if(str[0]=='a')
	  {
		n=atoi(str+1);
		k=n;
		l=1;
		while (k>0)
		{
			k=k/10;
			l++;
		}
		n--;

		i=atoi(str+l+2);
		k=i;
				while (k>0)
				{
					k=k/10;
					l++;
				}
		i--;
		j=atoi(str+l+3);
//		 printf("<%i,%i,%i> \n",n+1,i+1,j);
		if ((i<VER) && (n<VER))
		{
			d_tt=tt[n];
			 d_tt[i]=j;

	//		 puts(str);
	//		 printf("<%i,%i,%i> \n",n+1,i+1,j);
	//		 eds++;
		}
//		else printf("=======<%i,%i,%i> \n",n+1,i+1,j);
	  }
	}
	 fclose(f);
    puts("file was read");
// tt хранит десятичную матрицу весов на CPU, ее нужно преобразовать в бинарную на GPU.
//    действовать по столбцам
    int *d_tmp;
    LongPointer *d_tab;
    d_tab=T->get_device_pointer();

    cudaError_t err = cudaMalloc(&d_tmp,sizeof(int)*LENGTH1);

    for (j=0; j<M2; j++)
    {   d_tt=tt[j];
    	cudaMemcpy(d_tmp,tt[j],sizeof(int)*LENGTH1,cudaMemcpyHostToDevice);
    	printf("weight[%i][0]=%i \t",j,d_tt[0]);
 //   	printf("\n column %i\n",j);

    	vect_to_strip<<<H1,NN1>>>(j,d_tmp,d_tab);
    }
    err = cudaGetLastError();
    if (err!=0) printf("after vect_to_strip %d , %s \n",err,cudaGetErrorString(err));
// для матрицы Cost нужна транспонированная матрица tt
puts("weight");
    int *tc, *ttc;
    tc=new int[VER];
    for (j=0; j<M2; j++)
        {
    	for(i=0; i<M2;i++)
    		{ ttc=tt[i];
    		  tc[i]=ttc[j];
    		 if(j==0) printf("cost[0][%i]=%i \t",i,tc[i]);
    		}
        	cudaMemcpy(d_tmp,tc,sizeof(int)*LENGTH1,cudaMemcpyHostToDevice);
     //   	printf("\n column %i\n",j);
 //           puts(" --------- ");
        	vect_to_strip<<<H1,NN1>>>(j,d_tmp,d_tab);
        }
}
void initIO()
{
//	if(matrix==1)
	{
		size_tt=VER;
	}
//	else if (weighted==0)
//	{ size_tt=2;}
//	     else size_tt=3;

	tt=new int*[size_tt];
	for(int i=0; i<size_tt;i++) tt[i]=new int[LENGTH1+1];
}
