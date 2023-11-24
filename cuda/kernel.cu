#include <stdio.h>

typedef unsigned char uchar;

__host__
void errorexit(const char *s) {
    printf("\n%s\n",s);	
    exit(EXIT_FAILURE);	 	
}

__constant__ int Height;
__constant__ int Width;

__global__ void bicubic(uchar *bpixels, uchar *pixels){
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;
    extern __shared__ int suma[][3];

    if(row >= Height || col >= Width)
        return;

    int z = blockIdx.z;
    int index = z*Height*Width + row*Width + col;
    if(row == 0 || col == 0 || row == Height-1 || col == Width-1)
        bpixels[index] = pixels[z*Height*Width/4 + row/2*Width/2 + col/2];
    else{
        int sindex = + threadIdx.x*blockDim.y + threadIdx.y;
        suma[sindex][threadIdx.z] = 
            pixels[z*Height*Width/4 + (row - 1)/2*Width/2 + (col + threadIdx.z - 1)/2] +
            pixels[z*Height*Width/4 + row/2*Width/2 + (col + threadIdx.z - 1)/2] + 
            pixels[z*Height*Width/4 + (row + 1)/2*Width/2 + (col + threadIdx.z - 1)/2];

        __syncthreads();
        if(threadIdx.z == 0){
            int s = 0;
            for(int i = 0; i < 3; i++){
                s += suma[sindex][i];
            }
            bpixels[index] = (uchar)(s / 9);
        }
    }
}

void cudaFunction(uchar *hnpixels, uchar *hpixels, int height, int width) {
    
    int n = 2;

    int new_height = n * height;
    int new_width = n * width;

    long memory = height * width * 3 *sizeof(uchar);
    long memory2 = new_height * new_width * 3 *sizeof(uchar);

    uchar *dpixels;
    uchar *dbpixels;


    if(cudaSuccess != cudaMemcpyToSymbol(Height, &new_height, sizeof(int), 0, cudaMemcpyHostToDevice)){
        errorexit("Error copying `Height` on the GPU");
    }
    if(cudaSuccess != cudaMemcpyToSymbol(Width, &new_width, sizeof(int), 0, cudaMemcpyHostToDevice)){
        errorexit("Error copying `Width` on the GPU");
    }
    if (cudaSuccess != cudaMalloc((void**) &dpixels, memory)){
        errorexit("Error allocating `dpixels` memory on the GPU");
    }
    if (cudaSuccess != cudaMalloc((void**) &dbpixels,  memory2)){
        errorexit("Error allocating `dbpixels` memory on the GPU");
    }
    if (cudaSuccess!=cudaMemcpy(dpixels, hpixels, memory, cudaMemcpyHostToDevice)){
       errorexit("Error copying results from host to device");
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int nn = 8;
    dim3 blockSize(nn, nn, 3);
    dim3 gridSize(
        (new_height - 1)/ blockSize.x + 1,
        (new_width - 1)/ blockSize.y + 1, 
        3
    );

    cudaEventRecord(start);
        bicubic<<<gridSize, blockSize, nn * nn * 3 * sizeof(int)>>>(dbpixels, dpixels);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (cudaSuccess!=cudaMemcpy(hnpixels, dbpixels, memory2, cudaMemcpyDeviceToHost)){
       errorexit("Error copying results from device to host");
    }
    cudaFree(dpixels);
    cudaFree(dbpixels);
    printf("Time taken: %fms\n", milliseconds);
}