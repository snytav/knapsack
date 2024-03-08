
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

#include <cuda.h>
#include "star.h"
//#include "table.cu"
#include "basic.h"
#include "star_kernel.h"
#include "iostar.h"

__global__ void Warshall_kernel(LongPointer *d_tab,int k,unsigned long long int *d_x)
{       unsigned long long int *d_v,*d_w,w;
        int s;
//        int sizel = M;
//        double d_blocks;
//       unsigned int blocks, threads = M < SIZE_OF_LONG_INT ? M: SIZE_OF_LONG_INT;

//        d_blocks = M;
//        d_blocks = d_blocks/(double)threads;
//       	blocks = (sizel > SIZE_OF_LONG_INT)? (int)ceil( d_blocks) : 1;
//        get_row<<<blocks,threads>>>(d_tab,d_x,k,M);
	    d_w=d_tab[k-1];//tab->GetCol(w,k);
	    w=d_w[blockIdx.x];
    	s=get_position_bit(d_x,threadIdx.x+1);
    	if(s==1)
    	{
    		d_v=d_tab[threadIdx.x];//tab->GetCol(w,k);
    		d_v[blockIdx.x]|=w;

    	}
}

void warshall_c(Table *tab)
{
	Slice *X;//,*w,*v;
	unsigned long long int *d_x;
	LongPointer *d_tab;
	int k,NN;
    unsigned int lgs=tab->length;
    unsigned int sz=tab->size;
    X = new Slice;
  //  w = new Slice;
   // v = new Slice;

    X->Init(lgs);
    NN=X->NN;
  //  w->Init(sz);
  //  v->Init(sz);
    d_tab=tab->get_device_pointer();
    for (k = 1;k <= M;k++)
    {
    	tab->GetRow(X,k); //*X = tab->col(k);
    	d_x=X->get_device_pointer();
    	Warshall_kernel<<<NN,sz>>>(d_tab,k,d_x);
    }
}

__global__ void Warshall_ogr_kernel(LongPointer *d_tab,int k,int *n,unsigned long long int *d_x)
{       unsigned long long int *d_v,*d_w,w;
        int s;
//        int sizel = M;
//        double d_blocks;
//       unsigned int blocks, threads = M < SIZE_OF_LONG_INT ? M: SIZE_OF_LONG_INT;

//        d_blocks = M;
//        d_blocks = d_blocks/(double)threads;
//       	blocks = (sizel > SIZE_OF_LONG_INT)? (int)ceil( d_blocks) : 1;
//        get_row<<<blocks,threads>>>(d_tab,d_x,k,M);
	    d_w=d_tab[k-1];//tab->GetCol(w,k);
	    w=d_w[blockIdx.x];
    	s=get_position_bit(d_x,threadIdx.x+1);
    	if(s==1)
    	{
    		d_v=d_tab[threadIdx.x];//tab->GetCol(w,k);
    		d_v[blockIdx.x]|=w;
    		(*n)++;

    	}
}
void warshall_c_ogr(Table *tab)
{
	Slice *X;//,*w,*v;
	unsigned long long int *d_x;
	LongPointer *d_tab;
	int k,NN,n2,n1,n;
    unsigned int lgs=tab->length;
    unsigned int sz=tab->size;
    X = new Slice;
  //  w = new Slice;
   // v = new Slice;

    X->Init(lgs);
    NN=X->NN;
  //  w->Init(sz);
  //  v->Init(sz);
    d_tab=tab->get_device_pointer();
    n2=0; n1=1;
    for (k = 1;k <= M;k++)
    {   if(n2==n1)
        {k=M+1;}
        else
        {
        n=0;
        tab->GetRow(X,k); //*X = tab->col(k);
    	d_x=X->get_device_pointer();
    	Warshall_ogr_kernel<<<NN,sz>>>(d_tab,k,&n,d_x);
    	printf("k=%i,n=%i \n",k,n);
    	n2=n1;
    	n1=n;
        }
    }
}
__global__ void WarshallDev(LongPointer *d_tab, int k, unsigned long long int *d_x)
{	 unsigned long long int *d_v,*d_w;
	        int s;

		    d_w=d_tab[k-1];//tab->GetCol(w,k);
		    s=_get_bit(d_x,threadIdx.x+1);
	    	//s=get_position_bit(d_x,threadIdx.x+1);
	    	if(s==1)
	    	{
	    		d_v=d_tab[threadIdx.x];//tab->GetCol(w,k);
	    		_or(d_v,d_w);

	    	}
}
void warshall_c2(Table *tab)
{
	Slice *X;//,*w,*v;
	unsigned long long int *d_x;
	LongPointer *d_tab;
	int k,NN;
    unsigned int lgs=tab->length;
    unsigned int sz=tab->size;
    X = new Slice;
  //  w = new Slice;
   // v = new Slice;

    X->Init(lgs);
    NN=X->NN;
  //  w->Init(sz);
  //  v->Init(sz);
    d_tab=tab->get_device_pointer();
    for (k = 1;k <= M;k++)
    {
    	tab->GetRow(X,k); //*X = tab->col(k);
    	d_x=X->get_device_pointer();
    	WarshallDev<<<NN,sz>>>(d_tab,k,d_x);
    }
}

void warshall(Table *tab)
{
	Slice *X,*w,*v;
	int i,k;
    unsigned int lgs=tab->length;
    unsigned int sz=tab->size;
    X = new Slice;
    w = new Slice;
    v = new Slice;

    X->Init(lgs);
    w->Init(sz);
    v->Init(sz);

    for (k = 1;k <= M;k++)
    {
    	tab->GetRow(X,k); //*X = tab->col(k);
    	tab->GetCol(w,k);//*w = tab->row(k);
//    	X->print("X",0);
//    	w->print("w",1);
    	i = X->STEP();
    	while(i>0)
    	{
    		tab->GetCol(v,i);//*v = tab->row(i);
    		v->OR(w);
    		tab->SetCol(v,i);//*tab->row(i)=v;
    		i = X->STEP();
     	}
    }
}

    void warshall_o(Table *tab)
    {
    	Slice *X,*w,*v;
    	int i,k,n2,n1,n;
        unsigned int lgs=tab->length;
        unsigned int sz=tab->size;
        X = new Slice;
        w = new Slice;
        v = new Slice;

        X->Init(lgs);
        w->Init(sz);
        v->Init(sz);


        n2=M*M;
//        n1=1;
        n=0;
        for (k = 1;k <= M;k++)
        {
        	if(n==n2)
        	{
//        		printf("k=%i, n=%i \n",k,n);
        		k=M+1;
        	}
        	else
        	{
        	n=0;
        	tab->GetRow(X,k); //*X = tab->col(k);
        	tab->GetCol(w,k);//*w = tab->row(k);
    //    	X->print("X",0);
    //    	w->print("w",1);
        	i = X->STEP();
        	while(i>0)
        	{
        		tab->col(i)->OR(w);
  /*      		tab->GetCol(v,i);//*v = tab->row(i);
        		v->OR(w);
        		tab->SetCol(v,i);//*tab->row(i)=v;
   */
        		i = X->STEP();
//       		n=n+v->NUMB1();
        	}
 //       	n2=n1;
 //       	n1=n;
    //    	printf("k=%i, n=%i \n",k,n);
            }
        }
    }


__global__ void Triangles_kernel(LongPointer *d_tab,int k,unsigned long long int *d_x)
{       unsigned long long int *d_v,*d_w,w;
   //     int s;
   //     int sizel = M;
//        double d_blocks;
//       unsigned int blocks, threads = M < SIZE_OF_LONG_INT ? M: SIZE_OF_LONG_INT;

//        d_blocks = M;
//        d_blocks = d_blocks/(double)threads;
//       	blocks = (sizel > SIZE_OF_LONG_INT)? (int)ceil( d_blocks) : 1;
//        get_row<<<blocks,threads>>>(d_tab,d_x,k,M);
	    d_w=d_tab[k-1];//tab->GetCol(w,k);
	  _and(d_x,d_w);
}
__global__ void add_count(int *d_count, int *d_numb)
{
//	if (d_numb[0]>0)printf("%i+ %i\n",d_count[0],d_numb[0]);
	*d_count+=*d_numb;
}

__device__ void triangles_copy(LongPointer *d_tab, unsigned long long int *d_and, int i, unsigned long long int *d_res)
{ unsigned long long int *d_col;
  d_col=_col(d_tab,i-1);//d_tab[i];
  _assign(d_res,d_col);
  _and(d_res,d_and);
}
__global__ void triangles_kernel(LongPointer *d_tab, unsigned long long int *d_and, int i, unsigned long long int *d_res)
{ unsigned long long int *d_col;

  d_col=_col(d_tab,i-1);//d_tab[i];
  _assign(d_res,d_col);
  _and(d_res,d_and);
}
void CountTrianglesOPT(Table *tab, int * count)
{ // FILE *f;
  //  if((f = fopen("count_tr.dat","wt")) == NULL) return;
//	(*count)=0;
//	puts("countTr in");
	Slice *X,*Y,*mask;//,*w,*v;
	unsigned long long int *d_x,*d_y,*d_mask;
	LongPointer *d_tab;
	int m,k,j,NN,*d_count_t,* d_numb_x;
    unsigned int lgs=tab->length;
    unsigned int sz=tab->size;
    X = new Slice;
    Y = new Slice;
    mask= new Slice;
   // v = new Slice;

    cudaMalloc(&d_count_t,sizeof(int));
    cudaMemset(d_count_t,0,sizeof(int));
    cudaMalloc(&d_numb_x,sizeof(int));
    X->Init(lgs);
    Y->Init(lgs);
//    mask->Init(lgs);

    NN=X->NN;
  //  w->Init(sz);
  //  v->Init(sz);
    d_tab=tab->get_device_pointer();
    d_x=X->get_device_pointer();
    d_y=Y->get_device_pointer();
//    d_mask=mask->get_device_pointer();
//    puts("OPTIM::count init");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
 //   mask->SET();
    for (m=1; m<M;m++)
    {
 //   	triangles_kernel<<<blocks1,threads1>>>(d_tab,d_mask,m,d_x);
       tab->GetCol(X,m);
 ////   	mask->set(m,0);
//    	sprintf(s,"STEP_dat/mask%04d",m);
//    	mask->print(s,0);
 //   	count2=X->NUMB();
    	//if (count2>0)
//    	fprintf(f,"(%i) m=%i \n",count2,m);
    	j=X->STEP();
    	while(j>0)
    	{
//    	tab->GetCol(Y,j);
 //   	mask->MASK(j);
 //    	Y->AND(mask);
//    	mask->print("mask",0);
 //   	Y->AND(X);
  //  	triangles_kernel<<<blocks1,threads1>>>(d_tab,d_x,j,d_y); внесена в number_plus->copy_block_pluse
  //  	count1=Y->NUMB();//
//
    	number_plus(d_tab,d_x,j,d_y,NN,d_count_t,NN);
//    	count1=Y->NUMB();
//   	printf("m=%i,j=%i  \n",m,j);
 //   	add_count<<<1,1>>>(d_count_t,d_numb_x);
//    	(*count)+=count1;
    	//if (count1>0)
   // 	fprintf(f," count=%i: <%i,%i>:%i\n",*count,m,j,count1);
    	j=X->STEP();
    	}

    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime1, totalTime1;
    cudaEventElapsedTime(&elapsedTime1, start, stop);

    totalTime1 = elapsedTime1/(1000);

    printf("associative time count of triangles= %f\n", totalTime1);
//    fclose(f);
    cudaMemcpy(count,d_count_t,sizeof(int),cudaMemcpyDeviceToHost);
}

void CountTriangles(Table *tab, int * count)
{  FILE *f;
    if((f = fopen("count_tr.dat","wt")) == NULL) return;
	(*count)=0;
//	puts("countTr in");
	Slice *X,*Y,*mask;//,*w,*v;
	unsigned long long int *d_x;
	LongPointer *d_tab;
	int m,k,j,NN,*d_count_t,* d_numb_x,count1,count2=0;
    unsigned int lgs=tab->length;
    unsigned int sz=tab->size;
    X = new Slice;
    Y = new Slice;
    mask= new Slice;
   // v = new Slice;

    cudaMalloc(&d_count_t,sizeof(int));
    cudaMemset(d_count_t,0,sizeof(int));
    cudaMalloc(&d_numb_x,sizeof(int));
    X->Init(lgs);
    Y->Init(lgs);
    mask->Init(lgs);

    NN=X->NN;
  //  w->Init(sz);
  //  v->Init(sz);
    d_tab=tab->get_device_pointer();
    d_x=Y->get_device_pointer();
//    puts("count init");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (m=1; m<M;m++)
    {
    	tab->GetCol(X,m); //*X = tab->col(k);
 ////   	mask->MASK(m);
//    	if (m>(M-10))mask->print("mask",0);
  ////  	X->AND(mask);
 //   	count2=X->NUMB();
    	//if (count2>0)
//    	fprintf(f,"(%i) m=%i \n",count2,m);
    	j=X->STEP();
    	while(j>0)
    	{
    	tab->GetCol(Y,j);

 //   	mask->MASK(j);
 //    	Y->AND(mask);
//    	mask->print("mask",0);
    	Y->AND(X);
    	//Triangles_kernel<<<blocks1,threads1>>>(d_tab,j,d_x);
    	count1=Y->NUMB();//
  ////  	number(d_x,NN,d_numb_x,NN);

//    	printf("m=%i,j=%i  ",m,j);
//    	add_count<<<1,1>>>(d_count_t,d_numb_x);
    	(*count)+=count1;
    	if (count1>0)
    	fprintf(f," count=%i: <%i,%i>:%i\n",*count,m,j,count1);
    	j=X->STEP();
    	}

    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime1, totalTime1;
    cudaEventElapsedTime(&elapsedTime1, start, stop);

    totalTime1 = elapsedTime1/(1000);

    printf("associative time count of triangles= %f\n", totalTime1);
    fclose(f);
//    cudaMemcpy(count,d_count_t,sizeof(int),cudaMemcpyDeviceToHost);
}


__global__ void arcs(LongPointer *d_r, LongPointer *d_l, unsigned long long int *d_z, unsigned long long int *d_n,unsigned long long int *d_y, unsigned long long int *d_x ,int size)
{		//	    puts("vertex has nomber");
 match(d_r,d_z,d_n,d_y,size);
//   Z->print("Z",0);
//    Y->print("input arcs",0);
//X->print("X was",0);
 d_y[blockIdx.x]=~d_y[blockIdx.x];//Y->NOT();
//	Y->print("not_Y",0);
  d_x[blockIdx.x]&=d_y[blockIdx.x];  //X->AND(Y);
//	puts("the arcs were deleted");
//	X->print("search",0);

 match(d_l,d_x,d_n,d_y,size);
}


void DFS(Table *left, Table *right, Table *code, Slice *root, Table *NV, Slice *T, Slice *X)
{
	Table *LIFO;
	int lif,ord;
	int i;
	Slice *Y,*Z,*U,*V,*w,*node;

	LIFO=new Table;
	LIFO->Init(VER,M); // for saving code of vertices
	Y=new Slice;
	Y->Init(LENGTH1);
	Z=new Slice;
	Z->Init(LENGTH1);
	U=new Slice;
	U->Init(VER);
	V=new Slice;
	V->Init(VER);
	w=new Slice;
	w->Init(M);
	node=new Slice;
	node->Init(M);

	LongPointer *d_r, *d_l;
	unsigned long long int *d_x, *d_y, *d_z, *d_n;
    	int size=left->size;
	 d_x=X->get_device_pointer();
//	 d_n=node->get_device_pointer();
	 d_z=Z->get_device_pointer();
	 d_y=Y->get_device_pointer();
	 d_r=right->get_device_pointer();
	 d_l=left->get_device_pointer();
	 int NN;
	 NN=X->NN;

	Z->SET();
	T->CLR();
	U->SET();
	ord=1;
	lif=1;
	*node=*root;
	do
	{
	 //   node->print("node",0);
		LIFO->SetRow(node,lif);//*LIFO->row(lif)=node;
		 d_n=node->get_device_pointer();
//	puts("node is getting to stek");
		if(ord>VER)exit(0);
/*	    code->GetRow(w,ord);//*w=code->row(ord);
//	    w->print("dfs-number",0);
	    MATCH_CUDA(code,U,node,V);
//	    U->print("U",0);
	    i=V->FND();
//	    V->print("V",0);
//	    printf("ord= %d, lif= %d node %d \t", ord, lif,i);
	 //   printf("node %i %i \t", i, ord);
	    NV->SetRow(w,i);//*NV->row(i)=w;*/
		NV->SetRow(node,ord);//
	    ord++;
/*//	    puts("vertex has nomber");
	    MATCH_CUDA(right,Z,node,Y);
	 //   Z->print("Z",0);
	//    Y->print("input arcs",0);
	   //X->print("X was",0);
		Y->NOT();
	//	Y->print("not_Y",0);
		X->AND(Y);
	//	puts("the arcs were deleted");
	//	X->print("search",0);

		MATCH_CUDA(left,X,node,Y);
	//	X->print("X after MATCH",0);
	//	zero=Y->ZERO();
	//	puts("output arcs");
	//	Y->print("Y_before_Zero",0);
	*/
        arcs<<<NN,1>>>(d_r,d_l,d_z,d_n,d_y,d_x,size);

	    i=Y->FND();
//		printf("FND befor while %i \n",i);
		while((i==0)&&(lif>1))
		{
			lif--;
	//		printf("stek up %i \n",lif);
			LIFO->GetRow(node,lif);//*node=LIFO->row(lif);
		//	node->print("node_up",0);
		//	X->print("X in stec up",0);
			MATCH_CUDA(left,X,node,Y);
//			zero=Y->ZERO();
			i=Y->FND();
	//		printf(" up ord= %d, lif= %d \n", ord, lif);
		}

	//	X->print("X before stek down",0);
    //	Y->print("Y_before_Some",0);
	//	i=Y->FND();
//		printf("FND %i\n ",i);
		if (i>0)
		{	Y->set(i,0);
//		 puts("stek down");
//			 printf("right(%i)\n",i);
			 T->set(i,1);
			 lif++;
			 right->GetRow(node,i); //*node=right->row(i);
		//	 node->print("node_down",0);
	//		 printf("down ord= %d, lif= %d \n", ord, lif);
		}
		//X->print("X before next step",0);
		//T->print("T befor next step",0);
	}
	while(lif>1);
//	printf("ord %i", ord-1);
}

/*
 Procedure DFS(left, right: table; code: table; root: word;
              Var NV: table; Var T, X: slice);
Var LIFO: table; {моделирует стек [N,size(code)]}
    lif,{хранит глубину стека}
    ord, {хранит текущий М-номер}
    i: integer;
    Y, Z: slice{left} ;
    U, V: slice{code} ;
    w, node: word;
Begin
    SET(Z); CLR(T); SET(U);
    ord:=1; lif:=1; node:=root;
    repeat
       ROW(lif,LIFO):=node // Текущая вершина заносится в стек.
{Нумерация текущей вершины.}
       w:=ROW(ord, code);// двоичный код номера
       MATCH(code, U, node, V); i:=FND(V);// позиция вершины
       ROW(i, NV):=w; ord:=ord+1;
{Убираются дуги, ведущие в вершину node.}
       MATCH(right, Z, node, Y);
       Y:=not Y;
       X:=X and Y;
{Поиск дуг, ведущих из вершины node в ненумерованные вершины.}
       MATCH(left, X, node, Y);
{Если таких дуг нет, и текущая вершина не равна root, то поднимаемся по стеку.}
       while ZERO(Y) and (lif>1) do
       begin
          lif:=lif-1;
          node:=ROW(lif, LIFO);
          MATCH(left,X,node,Y);
       end;
{Если есть дуга из текущей вершины в ненумерованную, то заносим ее в дерево
и голова этой дуги становится текущей вершиной. Увеличивается глубина стека}
       if SOME(Y) then
       begin
         i:=STEP(Y);
         T(i):= 1;
         lif:=lif+1;
         node:=ROW(i, right);
       end;
   until lif=1;
End;
 */

void dijkstra1(Table *T,int s,Table *D)
{   int k,h=H1;
	Table *R1;//,*R2;
	LongPointer *t_R1, *t_T,*t_D;//, *t_R2;
	R1= new Table;
	R1->Init(LENGTH1,h);
    t_R1=R1->get_device_pointer();

    t_T=T->get_device_pointer();
    t_D=D->get_device_pointer();

/*	R2= new Table;
	R2->Init(LENGTH1,h);
	t_R2=R2->get_device_pointer();
*/
	Slice *U,*X,*Z,*inf,*v;

	X=new Slice;
	X->Init(LENGTH1);

	Z=new Slice;
	Z->Init(LENGTH1);

	inf=new Slice;
	inf->Init(h);
//	printf("infinit length=%i, elements=%i\n", inf->length,inf->NN);
	inf->SET(); //0x7FFFFFFF

	v=new Slice;
	v->Init(h);

	U=new Slice;
	U->Init(LENGTH1);
	U->SET();
//U->print("Dejk_U1.dat",0);
	U->set(s,0);
//U->print("Dejk_U2.dat",0);
	k=s;
	WCOPY(inf,U,D,H1);
	D->writeToFile("D0.dat");
//	puts("t_D");
//	 printStrip(t_D,0);
/*
	puts("t_T");
	for(int i=0; i<VER; i++)
	{
	 printf("strip %i\n",i);
	 printStrip(t_T,i);
	}
puts("T before copy R3.dat");
//writetoDimageW("graph10Dijk.dat",T);
//T->writeToFile("R3.dat");
//printStrip(t_T,k);
/*/
U->print("Dejk_U.dat",0);
	while (U->SOME())
	{
//printf("vertex %i\n",k);
		TCOPY1(T,k,h,R1); //копирует полосу с номером k шириной h
		R1->writeToFile("R1-1000_bin.dat");
printf("t_R1 k=%i\n",k);
//printStrip(t_R1,0);
printStrip(t_T,k-1);
//		R1->writeToFile("R1.dat");
//printf("T after copy R2.dat and R4.dat ");
//		T->writeToFile("R2.dat");

//		 writetoDimageW("R4.dat",T);
		MATCH_CUDA(R1,U,inf,X);
X->print("X_MATCH.dat",0);
		X->XOR(U);
//		printf("length %i\t",X->length);
X->print("XxorU",0);
		if (U->SOME())
//		if (X->SOME())
		{
			D->GetRow(v,k);
			printf("row(D,%i) ",k);
			v->print("v",1);
			//ADDC(R1,v,X,R2);
			ADDC1(X,v,R1);
//puts("R1+v");
//printStrip(t_R1,0);
			SETMIN(R1,D,X,Z);
			TMERGE(R1,Z,D);
            Z->print("opt_D",0);
puts("###############################new D");
printStrip(t_D,0);
		}
//U->print("U_before_MIN",0);
		MIN(D,U,X);
		X->print("X_MIN.dat",1);
		k=X->FND();
		U->set(k,0);
		U->print("U", 0);
	}
//		writetoDimageW("graph10Dijk.dat",T);
//  		T->writeToFile("graph10Dijk_bin.dat");
}

void dijkstra2(Table *T,int s,Table *D)
{   int k,h=H1;
	Table *R1;//,*R2;
	LongPointer *t_R1, *t_T,*t_D;//, *t_R2;
cudaError_t err = cudaGetLastError();
printf("before init dijkstra2 %d , %s \n",err,cudaGetErrorString(err));
	R1= new Table;
	R1->Init(LENGTH1,h);
    t_R1=R1->get_device_pointer();

    t_T=T->get_device_pointer();
    t_D=D->get_device_pointer();

/*	R2= new Table;
	R2->Init(LENGTH1,h);
	t_R2=R2->get_device_pointer();
*/
	Slice *U,*X,*Z,*P,*inf,*v;

	X=new Slice;
	X->Init(LENGTH1);

	Z=new Slice;
	Z->Init(LENGTH1);

	P=new Slice;
	P->Init(LENGTH1);

	inf=new Slice;
	inf->Init(h);
//	printf("infinit length=%i, elements=%i\n", inf->length,inf->NN);
	inf->SET(); //0x7FFFFFFF

	v=new Slice;
	v->Init(h);

	U=new Slice;
	U->Init(LENGTH1);
	U->SET();
//U->print("Dejk_U1.dat",0);
	U->set(s,0);
	P->set(s,1);
//U->print("Dejk_U2.dat",0);
	k=s;
	WCOPY(inf,U,D,H1);
//	D->writeToFile("D0.dat");
//	puts("t_D");
//	 printStrip(t_D,0);
/*
	puts("t_T");
	for(int i=0; i<VER; i++)
	{
	 printf("strip %i\n",i);
	 printStrip(t_T,i);
	}
puts("T before copy R3.dat");
//writetoDimageW("graph10Dijk.dat",T);
//T->writeToFile("R3.dat");
//printStrip(t_T,k);
/*/
//U->print("Dejk_U.dat",0);
	while (P->SOME())
	{
//printf("vertex %i\n",k);
		TCOPY1(T,k,h,R1); //копирует полосу с номером k шириной h
//		R1->writeToFile("R1-1000_bin.dat");
//printf("t_R1 k=%i\n",k);
//printStrip(t_R1,0);
//printStrip(t_T,k-1);
//		R1->writeToFile("R1.dat");
//printf("T after copy R2.dat and R4.dat ");
//		T->writeToFile("R2.dat");

//		 writetoDimageW("R4.dat",T);
		MATCH_CUDA(R1,U,inf,X);
//X->print("X_MATCH.dat",0);
		X->XOR(U);
//		printf("length %i\t",X->length);
//X->print("XxorU",0);
//		if (U->SOME())
		if (X->SOME())
		{
			D->GetRow(v,k);
//			printf("row(D,%i) ",k);
//			v->print("v",1);
			//ADDC(R1,v,X,R2);
			ADDC1(X,v,R1);
//puts("R1+v");
//printStrip(t_R1,0);
			SETMIN(R1,D,X,Z);
			TMERGE(R1,Z,D);
// Z->print("opt_D",0);
            P->OR(Z);
//puts("###############################new D");
//printStrip(t_D,0);
		}
//P->print("P_before_MIN",0);
		MIN(D,P,X);
//X->print("X_MIN.dat",1);
		k=X->FND();
//		U->set(k,0);
		P->set(k,0);

//err = cudaGetLastError();
//printf("before init dijkstra2 %d , %s \n",err,cudaGetErrorString(err));
//U->print("U", 0);
	}
//		writetoDimageW("graph10Dijk.dat",T);
//  		T->writeToFile("graph10Dijk_bin.dat");
}

__global__ void dijkstra_opt1(LongPointer *d_T,int k, int h, LongPointer *d_R1,unsigned long long int *d_U,unsigned long long int *d_inf,unsigned long long int *d_X)
{
//(T,k,h,R1,U,inf,X)
	// (LongPointer *d_t, int j, int h, LongPointer *d_f)
	tcopy1(d_T,k,h,d_R1);
	match(d_R1,d_U,d_inf,d_X,h);
	_xor(d_X,d_U);
}

__global__ void dijkstra_opt2(unsigned long long int *d_X,unsigned long long int *d_v,int h,LongPointer *d_R1,LongPointer *d_D,unsigned long long int *d_Z,unsigned long long int *d_P)
{
//	(X,v,R1,D,Z,P)
    			addc1(d_X,d_v,h,d_R1,d_Z);
				setmin(d_R1,d_D,d_X,d_Z);
				tmerge(d_R1,d_Z,d_D);
	            _or(d_P,d_Z);
				d_P[blockIdx.x]|=d_Z[blockIdx.x];
}

void dijkstra2_opt(Table *T,int s,Table *D)
{   int k,h=H1;
	Table *R1;//,*R2;
	LongPointer *t_R1, *t_T,*t_D;//, *t_R2;
	unsigned long long int *t_U,*t_X,*t_inf,*t_Z,*t_P,*t_v;
//cudaError_t err = cudaGetLastError();
//printf("before init dijkstra2_opt %d , %s \n",err,cudaGetErrorString(err));

R1= new Table;
	R1->Init(LENGTH1,h);
    t_R1=R1->get_device_pointer();

    t_T=T->get_device_pointer();
    t_D=D->get_device_pointer();

	Slice *U,*X,*Z,*P,*inf,*v;

	X=new Slice;
	X->Init(LENGTH1);
	t_X=X->get_device_pointer();

	Z=new Slice;
	Z->Init(LENGTH1);
    t_Z=Z->get_device_pointer();

	P=new Slice;
	P->Init(LENGTH1);
	t_P=P->get_device_pointer();

	inf=new Slice;
	inf->Init(h);
	t_inf=inf->get_device_pointer();
//	printf("infinit length=%i, elements=%i\n", inf->length,inf->NN);
	inf->SET(); //0x7FFFFFFF

	v=new Slice;
	v->Init(h);
    t_v=v->get_device_pointer();

	U=new Slice;
	U->Init(LENGTH1);
	t_U=U->get_device_pointer();
	U->SET();
//U->print("Dejk_U1.dat",0);
	U->set(s,0);
	P->set(s,1);
//U->print("Dejk_U2.dat",0);
	k=s;
//err = cudaGetLastError();
//printf("before dijkstra2_opt1 %d , %s \n",err,cudaGetErrorString(err));
	WCOPY(inf,U,D,H1);
	while (P->SOME())
	{
/* ---dijkstra_opt1(T,k,h,R1,U,inf,X)
		TCOPY1(T,k,h,R1); //копирует полосу с номером k шириной h
		MATCH_CUDA(R1,U,inf,X);
		X->XOR(U);
*/
		dijkstra_opt1<<<NN1,1>>>(t_T,k,h,t_R1,t_U,t_inf,t_X);
//		puts("opt1 ");
//		X->print("X_2",0);
//err = cudaGetLastError();
//printf("after dijkstra_opt1 %d , %s \n",err,cudaGetErrorString(err));
//		if (U->SOME())
		if (X->SOME())
		{
			D->GetRow(v,k);

//v->print("v",0);
//            ADDC1(X,v,R1);
			dijkstra_opt2<<<NN1,1>>>(t_X,t_v,h,t_R1,t_D,t_Z,t_P);
//			puts("opt2 ");
//err = cudaGetLastError();
//printf("after dijkstra_opt2 %d , %s \n",err,cudaGetErrorString(err));
/*///	---dijkstra_opt2(X,v,R1,D,Z,P)
			ADDC1(X,v,R1);
			SETMIN(R1,D,X,Z);
			TMARGE(R1,Z,D);
            P->OR(Z);
*/
		}
//		err = cudaGetLastError();
//		printf("before print %d , %s \n",err,cudaGetErrorString(err));
//	P->print("P_opt",0);
//	err = cudaGetLastError();
//	printf("after print %d , %s \n",err,cudaGetErrorString(err));
		MIN(D,P,X);
//X->print("X_opt_MIN",0);
//err = cudaGetLastError();
//printf("after MIN %d , %s \n",err,cudaGetErrorString(err));
		k=X->FND();
//err = cudaGetLastError();
//printf("after FND %d , %s \n",err,cudaGetErrorString(err));
//		U->set(k,0);
//printf("k=%i\n",k);
		P->set(k,0);
//		P->print("P_af_set0",0);
	}
}
