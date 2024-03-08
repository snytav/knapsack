#include "find.h"
#include "slice.h"
#include "table.h"
#include "star.h"
#include "basic.h"
//#include <sys/time.h>
#include <stdlib.h>

#include <stdio.h>
#include "basic-non.h"
#include "star_kernel.h"
#include "iostar.h"
#include "knapsack.h"

//#define rr
int warshall_test()
{
     Table *tab;
	 tab = new Table;
	 int eds=0;
     char fin[]="data/RG5000-4.dat";
     char fout[50];
     sprintf(fout,"res/Warshall/res_%i_w.dat",LENGTH1);
	 tab->Init(VER,VER); // matrix of adjacency
	 tab->readFromFileListAd_or(fin,& eds);
	 puts("file was read");
	 FILE *f_out;
	 double tt;
	 if((f_out=fopen(fout,"wt"))==NULL)return 0;
	 fprintf(f_out, "graph: |V|= %d |E|= %d \n", M,eds);
	 printf( "graph: |V|= %d |E|= %d \n", M,eds);
//	 struct timeval tv1,tv2,tv3;
	 	 	 cudaEvent_t start, stop;
	 	     cudaEventCreate(&start);
	 	     cudaEventCreate(&stop);
	 	    float elapsedTime1, totalTime1;
/*
     gettimeofday(&tv1,NULL);
	 warshall(tab);
     gettimeofday(&tv2,NULL);
	 tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
	 fprintf(f_out, "time of work warshall \t \t %f sec \n", tt);
	 printf("time of work warshall-C \t \t %f sec \n", tt);
	 puts("warshall done");
	 tab->writeToFile("res/Warshall/res_warshall-c.dat");
//	 puts("file res_warshall.dat was writen");
*/

	 tab->readFromFileListAd_or(fin,& eds);
	 puts("file was read");
//	 gettimeofday(&tv2,NULL);
	 cudaEventRecord(start, 0);
	 warshall_o(tab);
	 cudaEventRecord(stop, 0);
	 cudaEventSynchronize(stop);
		      	     cudaEventElapsedTime(&elapsedTime1, start, stop);// in 0.001 sec
		      	     tt = elapsedTime1/(1000);
//	 gettimeofday(&tv3,NULL); //
	 tab->writeToFile("res/Warshall/res_warshall-o.dat");
//     tt=0.000001*(tv3.tv_usec-tv2.tv_usec)+(tv3.tv_sec-tv2.tv_sec);
     fprintf(f_out, "time of work warshall_CUDA \t %f sec \n", tt);
     printf("time of work warshall_CUDA \t %f sec \n", tt);
     puts("warshal_o done");

     tab->readFromFileListAd_or(fin,& eds);
     puts("file was read");
//     	 gettimeofday(&tv2,NULL);
     	cudaEventRecord(start, 0);
     	 warshall_c(tab);
//     	 gettimeofday(&tv3,NULL);
     	 cudaEventRecord(stop, 0);
     	cudaEventSynchronize(stop);
     	cudaEventElapsedTime(&elapsedTime1, start, stop);// in 0.001 sec
     	tt = elapsedTime1/(1000);
     tab->writeToFile("res/Warshall/res_warshall-adapt.dat");
 //         tt=0.000001*(tv3.tv_usec-tv2.tv_usec)+(tv3.tv_sec-tv2.tv_sec);
          fprintf(f_out, "time of work warshall_CUD2  \t %f sec \n", tt);
          printf("time of work warshall_CUD2  \t %f sec \n", tt);
          puts("warshal_c done");
     fclose(f_out);
/*
	 	cudaError_t err = cudaGetLastError();

        Slice *root;
	 	root=new Slice;
	 	root->Init(M);
        root->SET();
	 	dim3 threads2D(threads1,1);
	 	dim3 blocks2D(blocks1,M);
	     cudaEvent_t start, stop;
	  	printf("errors before associative %d\n",err);
	     cudaEventCreate(&start);
	     cudaEventCreate(&stop);
	     cudaEventRecord(start, 0);
	     char s[20];
	     int i_num=0,i_num1=0;
//	     if((f_out=fopen("STEP_dat/STEP.dat","wt"))==NULL)return 0;
	     for(int i=M; i>0;i--)
	     {
//	    	 printf("numb %i",i);
	 //   	 X=left->col(i);//i);
	//    	 root->MASK(i);
	    	 i_num=root->NUMB();
//	    	 printf("step %i",i);
	//    	 i_num1=root->NUMB1();
  //           sprintf(s,"STEP_dat/step%04d",i);//ToString[i],".dat"}];
//	    	 root->print(s,0);
	    	 i_num1=root->STEP();
//	    	if(i<1000)
//	    		fprintf(f_out,"NUMB_naiv(%i)=%i   %i \n", i, i_num,i_num1);
//	    	 printf("NUMB_naiv(%i)=%i   %i \n", i, i_num1);
	     }
//	     fclose(f_out);
/*	     for( int i=20;i<70;i++)
//	    	 subtv_kernel<<<blocks1,threads1>>>(d_tab1,d_tab,h,d_x,d_res,d_z);
//	     subtc1_kernel<<<blocks1,threads1>>>(d_tab1, d_x, d_w, d_res,h,d_z);
	     { //tab->GetCol(root,i);
	       root->MASK(i);
	       root->print("MASK",0);
	     }*/
/*
//	    	 tmarge_kernel<<<blocks1,threads1>>>(d_tab,d_x,d_res);
//	    	 tmarge_kernel<<<blocks2D,threads2D>>>(d_tab,d_x,d_res);
//		  SUBTV(right,left,X,result);
//		  for( int i=0;i<1000;i++) MIN(right,X,Y,Z);
//    	 gettimeofday(&tv2,NULL);

	     cudaEventRecord(stop, 0);
	     cudaEventSynchronize(stop);
	     float elapsedTime1, totalTime1;
	     cudaEventElapsedTime(&elapsedTime1, start, stop);

	     totalTime1 = elapsedTime1/(1000*M);

	     printf("associative time row= %f\n", totalTime1);

	err = cudaGetLastError();
	printf("errors after associative %d\n",err);
//		 NV->writeToFile("res_DFS.dat");
//		 X->print("res_less_great",0);
//  	     root->print("res_subtc1_word",1);
//		 Y->print("res_MIN1-1000_slice",0);
*/
/*
//////////////////////////////////////////////
	          int blockSize;   // The launch configurator returned block size
			  int minGridSize; // The minimum grid size needed to achieve the
			                   // maximum occupancy for a full device launch
			  int gridSize;    // The actual grid size needed, based on input size

			  cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
					  match_kernel, 0, 0);
			  printf("optimal minGS=%i BS=%i \n", minGridSize,blockSize);
//////////////////////////////////////////////
 *
 */
/*
		 clock_t t;
      t=clock();
      cudaEventRecord(start, 0);
      for( int i=0;i<M;i++)
     	 //MIN(right,X,Y);
//     	 addv_kernel<<<blocks1,threads1>>>(d_tab1,d_tab,h,d_x,d_res,d_z);
      {     tab->GetRow_opt(root,i);
      tab->GetRow(root,i);}
	     cudaEventRecord(stop, 0);
	     cudaEventSynchronize(stop);
	     cudaEventElapsedTime(&elapsedTime1, start, stop);

	     totalTime1 = elapsedTime1/(1000*M);

	     printf("associative time row_opt= %f\n", totalTime1);
*/

     return 1;
}

int DFS_test()
{

	 Table *left, *right, *code,*NV;
	 Slice *X, *Y,*root;
	 int i=1; // number of root;

	 left = new Table;
	 right=new Table;
	 code=new Table;
	 NV=new Table;
	 int eds=0;

	 left->Init(LENGTH1,M);
	 right->Init(LENGTH1,M);
	 code->Init(VER,M);
	 NV->Init(VER,M);

	 X=new Slice;
	 X->Init(LENGTH1);
	 X->SET();


	 Y=new Slice;
	 Y->Init(LENGTH1);


	 root=new Slice;
	 root->Init(M);

	 readFromFileListLR("graph10.dat",&eds,left,right);
	 char str[M+1];
	 for(int i=1; i<=VER;i++)
	    {
	    	long_to_binary1(i,str,M);
	    	*root=str;
	    	code->SetRow(root,i);
	    }
     puts("code end");
	 code->GetRow(root,i);
	 FILE *f_out;
	 double tt;
//	 struct timeval tv1,tv2;
	 if((f_out=fopen("res_info_DFS-100000.dat","wt"))==NULL)return 0;
		 fprintf(f_out, "graph: |V|= %d |E|= %d \n", VER,LENGTH1);

//	 gettimeofday(&tv1,NULL);
	 puts("before DFS");
	 DFS(left, right, code, root, NV, Y, X);
//	 gettimeofday(&tv2,NULL);
	 NV->writeToFile("res_DFS.dat");
	 X->print("res_DFS_nnum",0);
	 Y->print("res_DFS_Tree",0);
//     tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
     fprintf(f_out, "time of work DFS %f sec \n", tt);
     fclose(f_out);

     return 1;
}

int LibraryTest()
{ Table *left, *right,*result;
Slice *X, *Y, *Z,*root;

left = new Table;
right=new Table;
result= new Table;
int eds=0;

left->Init(LENGTH1,M);
right->Init(LENGTH1,M);
result->Init(LENGTH1,M);
puts("init was done");
X=new Slice;
X->Init(LENGTH1);
X->SET();

/*if (X->SOME()) puts("some");
else puts("zero");
if (X->SOME1()) puts("some1");
else puts("zero1");
X->CLR();
if (X->SOME()) puts("after X->CLR some");
else puts("zero");
if (X->SOME1()) puts("after X->CLR some1");
else puts("zero1");
*/

X->set(10,0);
Y=new Slice;
Y->Init(LENGTH1);
Z=new Slice;
Z->Init(LENGTH1);
root=new Slice;
root->Init(M);


char fin[]="data/test_graph10000.txt";
    char fout[50];
    sprintf(fout,"res/test_lib/res_%i_%i_star.dat",LENGTH1,H1);
	 FILE *f_out;
	 if((f_out=fopen(fout,"wt"))==NULL)return 0;
	 initIO();
	 readfromDimageL(fin,left,right);
left->GetRow(root,10);

//puts("data was read");
unsigned long long int *d_x,*d_z, *d_w;
	 LongPointer *d_tab,*d_tab1,*d_res;
     int Nl=left->size;
     int NN=X->NN,i_num,i_num1;
     int h=H1;
     int LOOPS=100;

     d_x=X->get_device_pointer();
	 d_w=root->get_device_pointer();
	 d_z=Y->get_device_pointer();
	 d_tab=left->get_device_pointer();
	 d_tab1=right->get_device_pointer();
	 d_res=result->get_device_pointer();
//	 root->print("res_wmerge_word-before",1);
//	 struct timeval tv1,tv2;
//	 double tt;
/*	 if((f_out=fopen("res_min_time-100000-16.dat","wt"))==NULL)return 0;
	 fprintf(f_out, "graph: |V|= %d |E|= %d ,blocks=%d, threads=%d \n", VER,LENGTH1,blocks1,threads1);
	  int h=M;
       printf("Length=%i NN=%i, blocks=%i, threads=%i \n",LENGTH1, NN1, blocks1, threads1);
//	  gettimeofday(&tv1,NULL);
//	  MAX(left,X,Y);
*/

//	 addc1_kernel<<<NN,1>>>(d_x,d_w,h,d_tab,d_z);
        X->SET();
	 	cudaError_t err = cudaGetLastError();
	 	printf("errors before associative %d\n",err);

	 	dim3 threads2D(threads1,1);
	 	dim3 blocks2D(blocks1,M);
	     cudaEvent_t start, stop;
	     cudaEventCreate(&start);
	     cudaEventCreate(&stop);
//	     cudaEventRecord(start, 0);
	     float elapsedTime1, totalTime1;
/*	     for(int i=M; i>0;i--)
	     {
	 //   	 X=left->col(i);//i);
	    	 i_num=X->NUMB();
//	    	 i_num1=X->NUMB1();
	    	 X->STEP();
	    	 fprintf(f_out,"NUMB_naiv(%i)=%i   %i \n", i, i_num,i_num1);
	     }
*/

	     cudaEventRecord(start, 0);
	   	     for( int i=0;i<LOOPS;i++)
	   	    	 MATCH_CUDA(left,X,root,Z);
	   	     cudaEventRecord(stop, 0);
	   	     cudaEventSynchronize(stop);
	      	     cudaEventElapsedTime(&elapsedTime1, start, stop);// in 0.001 sec
	      	     totalTime1 = elapsedTime1/(1000*LOOPS);
	      	     fprintf(f_out,"associative time MATCH= %f (%i)\n", totalTime1, NN1);

	     cudaEventRecord(start, 0);
	     for( int i=0;i<LOOPS;i++)
	    	 MIN(right,X,Y,Z);
	     cudaEventRecord(stop, 0);
	     cudaEventSynchronize(stop);
   	     cudaEventElapsedTime(&elapsedTime1, start, stop);// в милисекундах!
   	     totalTime1 = elapsedTime1/(1000*LOOPS);
   	     i_num=Z->FND();
   	     fprintf(f_out,"associative time MIN= %f (%i)\n", totalTime1,i_num);


	     cudaEventRecord(start, 0);
//	     for( int i=0;i<1000;i++)
//                                             (LongPointer *d_t, LongPointer *d_r,int k, unsigned long long int *d_x, LongPointer *d_s,unsigned long long int *d_m)

	    	 subtv_kernel<<<blocks1,threads1>>>(d_tab1,           d_tab,               h,                         d_x,            d_res,                        d_z);
	     cudaEventRecord(stop, 0);
	     cudaEventSynchronize(stop);
   	     cudaEventElapsedTime(&elapsedTime1, start, stop);// в милисекундах!
   	     totalTime1 = elapsedTime1/1000;
   	     fprintf(f_out,"associative time SUBTV= %f (%i)\n", totalTime1,NN1);

	     cudaEventRecord(start, 0);
	     for( int i=0;i<LOOPS;i++)
	    	 tmerge_kernel<<<blocks1,threads1>>>(d_tab,d_x,d_res);
	     cudaEventRecord(stop, 0);
	     cudaEventSynchronize(stop);
   	     cudaEventElapsedTime(&elapsedTime1, start, stop);// в милисекундах!
   	     totalTime1 = elapsedTime1/(1000*LOOPS);
   	     fprintf(f_out,"associative time TMARCH1D= %f (%i)\n", totalTime1,NN1);

	     cudaEventRecord(start, 0);
	     for( int i=0;i<LOOPS;i++)
	    	 tmerge_kernel<<<blocks2D,H1>>>(d_tab,d_x,d_res);
	     cudaEventRecord(stop, 0);
	     cudaEventSynchronize(stop);
   	     cudaEventElapsedTime(&elapsedTime1, start, stop);// в милисекундах!
   	     totalTime1 = elapsedTime1/(1000*LOOPS);//
   	     fprintf(f_out,"associative time TMARCH2D= %f (%i)\n", totalTime1,NN1);

//	    	 subtv_kernel<<<blocks1,threads1>>>(d_tab1,d_tab,h,d_x,d_res,d_z);
//	     subtc1_kernel<<<blocks1,threads1>>>(d_tab1, d_x, d_w, d_res,h,d_z);
//	     left->GetRow(root,i);
//	    	 tmarge_kernel<<<blocks1,threads1>>>(d_tab,d_x,d_res);
//	    	 tmarge_kernel<<<blocks2D,threads2D>>>(d_tab,d_x,d_res);
//		  SUBTV(right,left,X,result);
//		  for( int i=0;i<1000;i++) MIN(right,X,Y,Z);
 //    	 gettimeofday(&tv2,NULL);

//	     cudaEventRecord(stop, 0);
//	     cudaEventSynchronize(stop);
//	     float elapsedTime1, totalTime1;
//	     cudaEventElapsedTime(&elapsedTime1, start, stop);
//	     totalTime1 = elapsedTime1/(1000*M);
//	     fprintf(f,"associative time NUMB= %f (%i)\n", totalTime1,i_num);
/*
 	err = cudaGetLastError();
 	printf("errors after associative %d\n",err);
//		 NV->writeToFile("res_DFS.dat");
//		 X->print("res_less_great",0);
//  	     root->print("res_subtc1_word",1);
//		 Y->print("res_MIN1-1000_slice",0);
		 int j=Y->FND();
		 clock_t t;
         t=clock();
         */
 /*        cudaEventRecord(start, 0);
         for(int i=M; i>0;i--)
       	     {
       	    	 X=left->col(i);//i);
       	    	 i_num=X->NUMB1();
       	    	printf("NUMB1(%i)=%i \n", i, i_num);
       	     }
    //     for( int i=0;i<1000;i++)
    //    	MIN(right,X,Y);
   //     	 addv_kernel<<<blocks1,threads1>>>(d_tab1,d_tab,h,d_x,d_res,d_z);
   //      left->GetRow_opt(root,i);
	     cudaEventRecord(stop, 0);
	     cudaEventSynchronize(stop);
	     cudaEventElapsedTime(&elapsedTime1, start, stop);

	     totalTime1 = elapsedTime1/(1000*M);

	     printf("time NUMB_thrust= %f (%i)\n", totalTime1,i_num);
	     */
 /*
	     //       t=clock()-t;
 //        printf("+ time = %f\n", ((float)t)/(CLOCKS_PER_SEC*1000));
     	err = cudaGetLastError();
      	printf("errors after associative %d\n",err);
 //        Y->print("res_MIN-1000_slice",0);
 //        root->print("res_min_1000_word",1);
 //          left->writeToFile("res_tmarge-1000_left.dat");
 //  	     right->writeToFile("res_min-1000_right.dat");
//		 result->writeToFile("res_addv-10_result.dat");
//	     tt=((tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec)*1000000)*0.01;
//	     fprintf(f_out, "time of work + %f usec \n", totalTime1);
	     fclose(f_out);
//	     printf("min = %i t=%.2lf usec \n",j,tt);

	     cudaEventDestroy(start);
	     cudaEventDestroy(stop);
 *
 */
	     maintest();

}

int triangles_test()
{
	 Table *tab;
	 tab = new Table;
	 int eds=0;

	 tab->Init(VER,VER); // matrix of adjacency
	 tab->readFromFileListAd_unor("graph_n13.dat",& eds);
	 puts("graph was read");
//	 tab->writeToFile("res_TM.dat");
	 printf("count of verteces=%i, count of edges=%i\n",VER ,eds);
	 FILE *f_out;
	 double tt;
//	 if((f_out=fopen("res_info_w.dat","wt"))==NULL)return 0;
//	 fprintf(f_out, "graph: |V|= %d |E|= %d \n", M,eds);
//     puts("file was read");
//	 struct timeval tv1,tv2,tv3;
//     gettimeofday(&tv1,NULL);
     CountTriangles(tab, &eds);
//     gettimeofday(&tv2,NULL);
//	 tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
	 printf("time of work thriangles \t \t %f sec (%i)\n", tt,eds);
	 puts("countTriangles done");

//	 tab->writeToFile("res_warshall.dat");
	//    gettimeofday(&tv1,NULL);
	     CountTrianglesOPT(tab, &eds);
	//     gettimeofday(&tv2,NULL);
//		 tt=0.000001*(tv2.tv_usec-tv1.tv_usec)+(tv2.tv_sec-tv1.tv_sec);
		 printf("time of work optimizated thriangles \t \t %f sec (%i)\n", tt,eds);
		 puts("countTriangles done");

/*	 	cudaError_t err = cudaGetLastError();

	     cudaEvent_t start, stop;
	     cudaEventCreate(&start);
	     cudaEventCreate(&stop);
	     cudaEventRecord(start, 0);

	     cudaEventRecord(stop, 0);
	     cudaEventSynchronize(stop);
	     float elapsedTime1, totalTime1;
	     cudaEventElapsedTime(&elapsedTime1, start, stop);

	     totalTime1 = elapsedTime1/(1000*1000);

	     printf("associative time row= %f\n", totalTime1);
*/
     return 1;
}
  void GRinput_test()
  { Table *tab;
	 tab = new Table;
	  initIO();
	 tab->Init(LENGTH1,M); // matrix of weights
	 readfromDimageW("USA-road-d.BAY.gr", tab);
	 writetoDimageW("wights.dat", tab);

  }

  void DecOut_test()
  {
	  Table *left, *right,*weight;
	  initIO();
	  left = new Table;
	  right=new Table;
	  weight=new Table;
	  int eds=0;

	  left->Init(LENGTH1,M);
	  right->Init(LENGTH1,M);
	  weight->Init(LENGTH1,M);

	  puts("init was done");

	  LongPointer *d_left;
	  d_left=left->get_device_pointer();
//	  readFromFileListLR("graph1000.dat",&eds,left,right);
	  readfromDimageL("USA-road-d.BAY.gr", left, right,weight);
//	  writeStrip("left_dec1.dat",d_left,0);
	  writetoDimageL("output.dat",left,right,weight);
	  printf("Infinite %i ",INFINITE);
  }



int main(void)
{
	InitArrays();

	knapsack_exp();
//	printf("Length=%i N=%i blocks=%i threads=%i \n", LENGTH1, NN1, blocks1,threads1);
}
