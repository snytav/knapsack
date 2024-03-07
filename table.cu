#include "find.h"
#include "table.h"
#include "param.h"
#include <stdio.h>
//#include "cuPrintf.cu"

 extern char **tb;//[LENGTH1][M]![LENGTH1][LENGTH1+1];  //table transposed

//#define ttt
int Table::Init(unsigned int lg,unsigned int sz)
{
	slice_device_pointer_table = (LongPointer *)malloc(sz*sizeof(LongPointer));
	cudaMalloc(&d_slice_device_pointer_table,sz*sizeof(LongPointer));
    length=lg;
    size=sz;
//	puts("Table INIT");
	for(int i = 0;i < sz;i++)
	{
		table[i].Init(lg);
//		printf("%i: %p\n",i+1, table[i].get_device_pointer());
	}
//	puts("Table INIT-2");

	InitDevicePointerTable();
//	puts("Table INIT-3");

	return 0;
}

__global__ void printPointer(LongPointer *p)
{
	unsigned long long *u = p[threadIdx.x];

	printf("printPointer %u %p %llu\n",threadIdx.x,p[threadIdx.x],*u);
//	printf("%u %llu\n",threadIdx.x,p[threadIdx.x]);
}

void Table::InitDevicePointerTable()
{

	for(int i = 0;i < size;i++)
	{
#ifdef ttt
		printf("slice_device_pointer_table[i] *************** %d \n",i);
#endif
		slice_device_pointer_table[i] = table[i].get_device_pointer();
#ifdef ttt
		printf("slice_device_pointer_table[i] *************** %d \n",i);
		printf("slice_device_pointer_table[i] *************** %d %p\n",i,slice_device_pointer_table[i]);
#endif
	}
	cudaMalloc(&d_slice_device_pointer_table,size*sizeof(unsigned long long int *));
	cudaMemcpy(d_slice_device_pointer_table,slice_device_pointer_table,size*sizeof(unsigned long long int *),
			cudaMemcpyHostToDevice);
#ifdef ttt
//	cudaPrintfInit();
//	printPointer<<<1,64>>>(d_slice_device_pointer_table);
//	    cudaPrintfDisplay(stdout, true);
//	    cudaPrintfEnd();
//	printPointer<<<1,64>>>(d_slice_device_pointer_table);
#endif
}

__global__ void get_row(LongPointer *p,unsigned long long *d_v,int i,unsigned int size)
{
//	char s[100];
	 __shared__ unsigned long long  int tmp[SIZE_OF_LONG_INT];
//	cuPrintf("get_row \n");
//	return;

	 unsigned int n1 = threadIdx.x + blockIdx.x*blockDim.x;
	 tmp[threadIdx.x] = 0;
	 if(n1 >= size ) return;
//	 return;
	 unsigned long long *d_rhs = p[n1];
#ifdef ttt
	 cuPrintf("get_row1.5 %u %p \n",n1,p[n1]);
//	 return;
	 cuPrintf("get_row2 %u %p \n",n1,d_rhs);
//	 return;
#endif
//	long_to_binary(*d_rhs,s,SIZE_OF_LONG_INT);
//	printf("get_row %s\n",s);
//	return;
	//assign_bit(d_v,threadIdx.x,n,OR);
	unsigned long long int n = get_position_bit(d_rhs,i);
//	return;
//	cuPrintf("i %d blockIdx.x %d threadIdx.x %d n %d \n",i,blockIdx.x,threadIdx.x,n);
//    return;
	int ni;

	ni = blockIdx.x;//get_64bit_word(threadIdx.x,SIZE_OF_LONG_INT);
	d_v[ni] = 0;

	int pos = position_in_64bit_word(threadIdx.x,SIZE_OF_LONG_INT);
	unsigned long long int u = n << pos;
#ifdef ttt
	cuPrintf("threadIdx.x %d ni%d pos %d n << pos %d\n",threadIdx.x,ni,pos,(int)u);
#endif
	tmp[threadIdx.x] = u;
#ifdef ttt
	cuPrintf("threadIdx.x %d %d getarray %d \n",threadIdx.x,(int)(tmp[threadIdx.x]),
			(int)(get_array(tmp,threadIdx.x,SIZE_OF_LONG_INT)));
#endif
    int M1=blockDim.x;//SIZE_OF_LONG_INT;
//	printf("before %d %luu \n",ni, d_v[ni]);
    d_v[ni] =  get_array(tmp,0,M1)
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

#ifdef ttt
   printf("%d %d: %d %d get_array %llu \n",blockIdx.x,blockDim.x, threadIdx.x,ni, get_array(tmp,threadIdx.x,M1));
#endif
	//assign_bit(d_v,threadIdx.x,n,OR);
//	long_to_binary(d_v[ni],s,M);
//	printf("get_row_res %s \n",s);

}
__global__ void get_row_opt(LongPointer *p,unsigned long long *d_v,int i,unsigned int size)
{
//	char s[100];
	 __shared__ unsigned long long  int tmp[SIZE_OF_LONG_INT];
//	cuPrintf("get_row \n");
//	return;
     unsigned long long int tmp_half[2];
	 unsigned int n1 = threadIdx.x + blockIdx.x*blockDim.x;
	 tmp[threadIdx.x] = 0;
	 if(n1 >= size ) return;
//	 return;
	 unsigned long long *d_rhs = p[n1];
#ifdef ttt
	 cuPrintf("get_row1.5 %u %p \n",n1,p[n1]);
//	 return;
	 cuPrintf("get_row2 %u %p \n",n1,d_rhs);
//	 return;
#endif
//	long_to_binary(*d_rhs,s,SIZE_OF_LONG_INT);
//	printf("get_row %s\n",s);
//	return;
	//assign_bit(d_v,threadIdx.x,n,OR);
	unsigned long long int n = get_position_bit(d_rhs,i);
//	return;
//	cuPrintf("i %d blockIdx.x %d threadIdx.x %d n %d \n",i,blockIdx.x,threadIdx.x,n);
//    return;
	int ni;

	ni = blockIdx.x;//get_64bit_word(threadIdx.x,SIZE_OF_LONG_INT);
	d_v[ni] = 0;

	int pos = position_in_64bit_word(threadIdx.x,SIZE_OF_LONG_INT);
	unsigned long long int u = n << pos;
#ifdef ttt
	cuPrintf("threadIdx.x %d ni%d pos %d n << pos %d\n",threadIdx.x,ni,pos,(int)u);
#endif
	tmp[threadIdx.x] = u;
#ifdef ttt
	cuPrintf("threadIdx.x %d %d getarray %d \n",threadIdx.x,(int)(tmp[threadIdx.x]),
			(int)(get_array(tmp,threadIdx.x,SIZE_OF_LONG_INT)));
#endif
    int M1=blockDim.x;//SIZE_OF_LONG_INT;
//	printf("before %d %luu \n",ni, d_v[ni]);
    if (threadIdx.x<32)
    		{tmp_half[0]=get_array(tmp,0,M1)
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
            |  get_array(tmp,31,M1);
    		}
    else
    {
    	tmp_half[1]=get_array(tmp,32,M1)
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
    }
    d_v[ni] =  tmp_half[0]|tmp_half[1];

#ifdef ttt
   printf("%d %d: %d %d get_array %llu \n",blockIdx.x,blockDim.x, threadIdx.x,ni, get_array(tmp,threadIdx.x,M1));
#endif
	//assign_bit(d_v,threadIdx.x,n,OR);
//	long_to_binary(d_v[ni],s,M);
//	printf("get_row_res %s \n",s);

}

Slice *Table::row(int i)
{
	Slice *s;
	unsigned int M2;
	unsigned long long *d_v;
//	puts("row");
//	i--;
    M2=size;
	s = new Slice();
	s->Init(size);
//	printf("col: length %u, %u \n",s->length,s->NN);
    d_v = s->get_device_pointer();
#ifdef ttt
    printf("row2 %p,%p\n",s,d_v);
#endif
    int sizel = size; //size;
    double d_blocks;
   unsigned int blocks, threads = M2 < SIZE_OF_LONG_INT ? M2: SIZE_OF_LONG_INT;

    d_blocks = M2;
    d_blocks = d_blocks/(double)threads;
   	blocks = (sizel > SIZE_OF_LONG_INT)? (int)ceil( d_blocks) : 1;
//   	printf("col: blocks %u; threads %u \n",blocks,threads);
#ifdef ttt
   	for(int i = 0;i < length;i++)
   	{
   		printf("d_slice_device_pointer_table %d %p \n",i,slice_device_pointer_table[i]);
   	}
//#ifdef ttt
   	cudaPrintfInit();
   	printPointer<<<1,64>>>(d_slice_device_pointer_table);
    cudaPrintfDisplay(stdout, true);
       cudaPrintfEnd();

   	cudaPrintfInit();
#endif
 //   int num,sh;
//     	num = get_64bit_word(i,SIZE_OF_LONG_INT);
 //  		sh =  position_in_64bit_word(i,SIZE_OF_LONG_INT);
//   	printf("num %d %d, sh %d, %d \n",i, num,sh,i%64);

   	get_row<<<blocks,threads>>>(d_slice_device_pointer_table,d_v,i,length);
#ifdef ttt
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
#endif
//#ifdef ttt
//    s->get_device_pointer()=d_v;
//     printf("col3 %p,%p\n",s,d_v);
//#endif
//    s->print("col-",0);
    return s;
}
void Table::GetRow(Slice *s,int i)
{
	unsigned int M2;
	unsigned long long *d_v;
//	puts("row");
//	i--;
    M2=size;
//	printf("col: length %u, %u \n",s->length,s->NN);
    d_v = s->get_device_pointer();
#ifdef ttt
    printf("row2 %p,%p\n",s,d_v);
#endif
    int sizel = size;
    double d_blocks;
   unsigned int blocks, threads = M2 < SIZE_OF_LONG_INT ? M2: SIZE_OF_LONG_INT;

    d_blocks = M2;
    d_blocks = d_blocks/(double)threads;
   	blocks = (sizel > SIZE_OF_LONG_INT)? (int)ceil( d_blocks) : 1;
 //  	printf("col: blocks %u; threads %u \n",blocks,threads);
#ifdef ttt
   	for(int i = 0;i < length;i++)
   	{
   		printf("d_slice_device_pointer_table %d %p \n",i,slice_device_pointer_table[i]);
   	}
//#ifdef ttt
   	cudaPrintfInit();
   	printPointer<<<1,64>>>(d_slice_device_pointer_table);
    cudaPrintfDisplay(stdout, true);
       cudaPrintfEnd();

   	cudaPrintfInit();
#endif
 //   int num,sh;
//     	num = get_64bit_word(i,SIZE_OF_LONG_INT);
 //  		sh =  position_in_64bit_word(i,SIZE_OF_LONG_INT);
//   	printf("num %d %d, sh %d, %d \n",i, num,sh,i%64);

   	get_row<<<blocks,threads>>>(d_slice_device_pointer_table,d_v,i,size);
#ifdef ttt
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
#endif
//#ifdef ttt
//    s->get_device_pointer()=d_v;
//     printf("col3 %p,%p\n",s,d_v);
//#endif
//    s->print("col-",0);
//    return s;
}
void Table::GetRow_opt(Slice *s,int i)
{
	unsigned int M2;
	unsigned long long *d_v;
//	puts("row");
//	i--;
    M2=size;
//	printf("col: length %u, %u \n",s->length,s->NN);
    d_v = s->get_device_pointer();
#ifdef ttt
    printf("row2 %p,%p\n",s,d_v);
#endif
    int sizel = size;
    double d_blocks;
   unsigned int blocks, threads = M2 < SIZE_OF_LONG_INT ? M2: SIZE_OF_LONG_INT;

    d_blocks = M2;
    d_blocks = d_blocks/(double)threads;
   	blocks = (sizel > SIZE_OF_LONG_INT)? (int)ceil( d_blocks) : 1;
 //  	printf("col: blocks %u; threads %u \n",blocks,threads);
#ifdef ttt
   	for(int i = 0;i < length;i++)
   	{
   		printf("d_slice_device_pointer_table %d %p \n",i,slice_device_pointer_table[i]);
   	}
//#ifdef ttt
   	cudaPrintfInit();
   	printPointer<<<1,64>>>(d_slice_device_pointer_table);
    cudaPrintfDisplay(stdout, true);
       cudaPrintfEnd();

   	cudaPrintfInit();
#endif
 //   int num,sh;
//     	num = get_64bit_word(i,SIZE_OF_LONG_INT);
 //  		sh =  position_in_64bit_word(i,SIZE_OF_LONG_INT);
//   	printf("num %d %d, sh %d, %d \n",i, num,sh,i%64);

   	get_row_opt<<<blocks,threads>>>(d_slice_device_pointer_table,d_v,i,size);
#ifdef ttt
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
#endif
//#ifdef ttt
//    s->get_device_pointer()=d_v;
//     printf("col3 %p,%p\n",s,d_v);
//#endif
//    s->print("col-",0);
//    return s;
}
void Table::SetCol(Slice *s,int i)
{   unsigned int NN=s->NN;
	unsigned long long int *d_v2,*d_v1;
	d_v2=slice_device_pointer_table[i-1];
	d_v1=s->get_device_pointer();
	set_kernel<<<NN,1>>>(d_v2,d_v1);
}
void Table::GetCol(Slice *s,int i)
{
		unsigned int NN=s->NN;
			unsigned long long int *d_v2,*d_v1;
			d_v2=slice_device_pointer_table[i-1];
			d_v1=s->get_device_pointer();
//			printf("GetRoW %u, row %p,s %p",NN,d_v2,d_v1);
			set_kernel<<<blocks1,threads1>>>(d_v1,d_v2);
}
void Table::readFromFile(char *fn)
{
//	char tb[M][LENGTH+1];  //table transposed
	char str[LENGTH1+1];
	FILE *f;
	int   n = 0;
    int M2=size;
	if((f = fopen(fn,"rt")) == NULL) return;

	while(fgets(str,2*M2,f) != NULL)
	{
		//puts(str);
//		printf("reading %d line ",n);
		for(int i = 0;i < M2;i++)
		{
			tb[n][i] = str[i];
//			printf("%c",tb[n][i]);
		}
//		printf("\n");
/*		for(int i = 0;i < length;i++)
		{
			tb[i][M2] = 0;
		}*/
		tb[n][M2] = 0;
		n++;
	}
	fclose(f);
//	printf("was read \n");
//	return;
	Slice *s;
	s=new Slice;
	s->Init(length);
	for(int i = 0;i < M2;i++)
	{

//		printf("copying %d slice ",i);
	for(int j = 0;j < length;j++)
		{
			str[j] = tb[j][i];
//			printf("%c",tb[j][i]);
		}
		str[LENGTH1] = 0;
//        printf("\n");

		printf("slice %d %s \n",i,str);
//		s = &(table[i]);
		*s = str;
		puts("tab");
		s->print("ss",0);
/*#ifdef ttt
		sprintf(fname,"before_slice%02d",i);
		s->print(fname,1);
		s->print(fname,0);


		unsigned long long int *d_v,*h_v;
		d_v = s->get_device_pointer();
        h_v = (unsigned long long *)malloc(N*sizeof(unsigned long long));
		cudaError_t err = cudaMemcpy(h_v,d_v,N*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
//        printf("string from slice ");
//
//		for(int j = 0;j < N;j++)
//		{
//            long_to_binary(h_v[j],str);
//            printf("%s",str);
//		}
//		printf("\n");
#endif */


//		sprintf(fname,"slice%02d",i);
//		s->print(fname,1);
//		s->print(fname,0);
	}
#ifdef tttt
    puts("end read from file\n");
#endif
}
void Table::readFromFileListAd_or(char *fn,int *eds)
{
	int ras = 100;// max(100, 2 * VER);
	char str[100];
	FILE *f;
	int   n,i,k,l;
	int M2=size;
	*eds=0;

//	puts(" begin set by 0\n");
    for (l=0; l<M2;l++)
    {
    for(int i = 0;i < length;i++)
	      {
	      	tb[i][l] ='0';
	      }
    }
	if((f = fopen(fn,"rt")) == NULL) return;
//	puts("set by 0\n");
	while(fgets(str,ras,f) != NULL)
	{
//		puts(str);
/*
		for(int i = 0;i < M;i++)
		{
			tb[n][i] = str[i];
//			printf("%c",tb[n][i]);
		}
//		printf("\n");
		for(int i = 0;i < LENGTH;i++)
		{
			tb[i][M] = 0;
		}
		n++;
*/

      if(str[0]!='#')
      {
		n=atoi(str);
		k=n;
		l=1;
		while (k>0)
		{
			k=k/10;
			l++;
		}
		n--;
		i=atoi(str+l);
		i--;
//		printf("<%i,%i> \n",n,i);
		if ((i<M2) && (n<M2)){
			tb[n][i]='1';
			(*eds)++;
		}
      }
      for(int i = 0;i < length;i++)
      {
      	tb[i][M2] = 0;
      }
//      puts("-");
	}
	fclose(f);
//	puts("set by  list\n");
//	printf("\n");
//	return;
	Slice *s;
	s=new Slice;
	s->Init(size);
	for(int i = 0;i < size;i++)
	{
//		printf("copying %d slice ",i);
	for(int j = 0;j < length;j++)
		{
			str[j] = tb[i][j];
//			printf("%c",tb[j][i]);
		}
//		str[M] = 0;
//        printf("\n");

//		printf("slice %d %s \n",i,str);
		s = &(table[i]);
		*s = str;
//		puts(str);
/*#ifdef ttt
		sprintf(fname,"before_slice%02d",i);
		s->print(fname,1);
		s->print(fname,0);


		unsigned long long int *d_v,*h_v;
		d_v = s->get_device_pointer();
        h_v = (unsigned long long *)malloc(N*sizeof(unsigned long long));
		cudaError_t err = cudaMemcpy(h_v,d_v,N*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
#endif */

//		s->print("read",0);
	}
//    puts("end read from file\n");
}

void Table::readFromFileListAd_unor(char *fn,int *eds)
{

	char str[2*VER];
	FILE *f;
	int   n,i,k,l;
	int M2=size;
	*eds=0;

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
	puts("set by 0\n");
	while(fgets(str,2*M2,f) != NULL)
	{
/*
		//puts(str);

		for(int i = 0;i < M;i++)
		{
			tb[n][i] = str[i];
//			printf("%c",tb[n][i]);
		}
//		printf("\n");
		for(int i = 0;i < LENGTH;i++)
		{
			tb[i][M] = 0;
		}
		n++;
*/

      if((str[0]!='#')&(str[0]!='%'))
      {
		n=atoi(str);
		k=n;
		l=1;
		while (k>0)
		{
			k=k/10;
			l++;
		}
		n--;
		i=atoi(str+l);
		i--;
		if ((i<M2) && (n<M2)){
			if (n<i) tb[n][i]='1';
			if (i<n) tb[i][n]='1';
			(*eds)++;
		}
      }
      for(int i = 0;i < length;i++)
      {
      	tb[i][M2] = 0;
      }
	}
	fclose(f);
//	puts("set by  list\n");
//	printf("\n");
//	return;
	Slice *s;
	s=new Slice;
	s->Init(size);
	for(int i = 0;i < size;i++)
	{
//		printf("copying %d slice ",i);
	for(int j = 0;j < length;j++)
		{
			str[j] = tb[i][j];
//			printf("%c",tb[j][i]);
		}
//		str[M] = 0;
//        printf("\n");

//		printf("slice %d %s \n",i,str);
		s = &(table[i]);
		*s = str;
//		puts(str);
/*#ifdef ttt
		sprintf(fname,"before_slice%02d",i);
		s->print(fname,1);
		s->print(fname,0);


		unsigned long long int *d_v,*h_v;
		d_v = s->get_device_pointer();
        h_v = (unsigned long long *)malloc(N*sizeof(unsigned long long));
		cudaError_t err = cudaMemcpy(h_v,d_v,N*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
#endif */

//		s->print("read",0);
	}
//    puts("end read from file\n");
}

void Table::writeToFile(char *fn)
{
//	char tb[M][LENGTH+1];  //table transposed
//	char str[NN1*64]; //LENGTH1+1 дает переполнение стека
	FILE *f;
	int   n = 0;

//	printf("write %d %d", length,M2);
	if((f = fopen(fn,"wt")) == NULL)return;
	Slice *s;
	 s=new Slice;
	 s->Init(size); //row
	for(n = 0;n < size;n++)
	{
		s = &(table[n]);
//		GetRow(s,n+1);
//   printf("n=%d",n);
//#ifdef tt
		s->convert_to_string(str);

//		s->print("after_slice",0);
//		s->print("after_slice",1);
//		puts(str);
//#endif
//		puts(str);
		for(int i = 0;i < length;i++)
		{
			tb[n][i] = str[i];
//			fprintf(f,"%c",tb[n][i]);
		}
//		fprintf(f,"\n");
	}
//    puts("table ended");
	for(n = 0;n < length;n++)
	{

		for(int i = 0;i < size;i++)
		{
			fprintf(f,"%c",tb[i][n]);
//			printf("<%i,%i>%c\t",i,n,tb[i][n]);
		}
		fprintf(f,"\n");
	}
	puts("print all");

	fclose(f);
	puts("file closed");
}

__global__ void set_row(LongPointer *p,int i,unsigned long long int *d_v, int size)
{
#ifdef QQ
	char s[100];
	long_to_binary(d_v[0],s);
	printf("s %s\n",s);
#endif
//	printf("qq\n");
//	return;
	int index=threadIdx.x + blockIdx.x*blockDim.x;//blockIdx.x
	if (index>size-1) return;
	unsigned long long *d_rhs = p[index];
	int n = get_position_bit(d_v,index+1);
//#ifdef ttt
//	printf("threadIdx.x %d %d n %d \n",blockIdx.x,i,n);
//#endif
//	long_to_binary(*d_rhs,s);

	assign_bit(d_rhs,i,n,SET);
}

void Table::SetRow(Slice *s,int i)
{
//	i--;

//	s->print("v_row_in",1);
	unsigned long long int *d_v = s->get_device_pointer();
	 cudaError_t err;
#ifdef ttt
   unsigned long long int *h_v;
   char str[100];

   h_v = (unsigned long long int *)malloc(N1*sizeof(unsigned long long int));

   err = cudaMemcpy(h_v,d_v,N1*sizeof(unsigned long long int),cudaMemcpyDeviceToHost);

   long_to_binary(h_v[0],str,length);
   printf("entering SetRow err %d %s %s pointer %p,%p\n",err,cudaGetErrorString(err),str,s,d_v);
#endif
    int size1 = size;
   unsigned int blocks, threads =size1< SIZE_OF_LONG_INT ? size1: SIZE_OF_LONG_INT;//1;



  	blocks =(size1 > SIZE_OF_LONG_INT)? (int)ceil( ((double)size1)/threads) : 1;//size1;
#ifdef ttt
    printf("SetRow %d %d i %d \n",blocks,threads,i);

    err = cudaGetLastError();

    printf("eerr %d %s\n",err,cudaGetErrorString(err));
#endif

    //slice addressing function again needs 1-based numbers
    set_row<<<blocks,threads>>>(d_slice_device_pointer_table,i,d_v,size1);
  //  printf("blocks %i, threads %i \n", blocks,threads);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
#ifdef ttt
    printf("eerr %d %s\n",err,cudaGetErrorString(err));
#endif
//   6 exit(0);
}

void readFromFileListLR(char *fn,int *eds, Table *left, Table *right)
{
	char str[50],sleft[65], sright[65];
	FILE *f;
	unsigned long long int   n,i;
	int k,j,num, edd;
	int M2=left->size;
	num=left->length;
	Slice *s;
	s =new Slice;
	s->Init(M);
	edd=1; //number of edges;

	//	puts(" begin set by 0\n");
/*
	for (l=0; l<M2;l++)
    {
    for(int i = 0;i < length;i++)
	      {
	      	tb[i][l] ='0';
	      }
    }
 */
	if((f = fopen(fn,"rt")) == NULL) return;
//	puts("set by 0\n");

	while((fgets(str,100,f) != NULL) && (edd<=num))
	{
      if(str[0]!='#')
      {
    	// puts(str);
		n=atoi(str);
		k=n;
		j=1;
		while (k>0)
		{
			k=k/10;
			j++;
		}
		i=atoi(str+j);
		if ((i<= VER) && (n<=VER))
		{
//			tb[i][n]='1';
			long_to_binary1(n,sleft,M2);
			long_to_binary1(i,sright,M2);
//			printf("left %d: %llu " ,edd, n);
//			puts(sleft);
//			printf("right %llu ",i);
//			puts(sright);

			*s=sleft;
			left->SetRow(s,edd);
			*s=sright;
			right->SetRow(s,edd);
			edd++;
			if(i==0)
				{
				printf("read %d %llu %llu\n",edd,n,i);
				exit(0);
				}
		}
      }
 /*     else
      {
    	  puts(str);
      }*/
	}
	*eds=edd;
	fclose(f);
//    puts("end read from file\n");

}
