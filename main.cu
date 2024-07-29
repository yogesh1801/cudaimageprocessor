#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

__global__ 
void 
rgb_to_grayscale(const uchar3* input, unsigned char* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        uchar3 rgb = input[idx];
        output[idx] = static_cast<unsigned char>(0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z);
    }
}

__global__ 
void 
box_blur(const unsigned char* input, unsigned char* output, int width, int height, int kernelSize) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int sum = 0;
        int count = 0;
        int halfKernel = kernelSize / 2;
        
        for (int dy = -halfKernel; dy <= halfKernel; dy++) {
            for (int dx = -halfKernel; dx <= halfKernel; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    sum += input[ny * width + nx];
                    count++;
                }
            }
        }
        
        output[y * width + x] = sum / count;
    }
}

int main() {
    cv::Mat input = cv::imread("img.png", cv::IMREAD_COLOR);
    if (input.empty()) {
        std::cerr << "Error: Could not read input image." << std::endl;
        return -1;
    }
    
    int width = input.cols;
    int height = input.rows;
    
    uchar3* d_input;
    unsigned char* d_grayscale;
    unsigned char* d_blurred1;
    unsigned char* d_blurred2;

    cudaMalloc(&d_input, width * height * sizeof(uchar3));
    cudaMalloc(&d_grayscale, width * height * sizeof(unsigned char));
    cudaMalloc(&d_blurred1, width * height * sizeof(unsigned char));
    cudaMalloc(&d_blurred2, width * height * sizeof(unsigned char));
    
    cudaMemcpy(d_input, input.data, width * height * sizeof(uchar3), cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    rgb_to_grayscale<<<gridSize, blockSize>>>(d_input, d_grayscale, width, height);
    
    int kernelSize = 9;  
    int numPasses = 5;  
    
    for (int i = 0; i < numPasses; i++) {
        if (i % 2 == 0) {
            box_blur<<<gridSize, blockSize>>>(i==0 ? d_grayscale:d_blurred2, d_blurred1, width, height, kernelSize);
        } else {
            box_blur<<<gridSize, blockSize>>>(d_blurred1, d_blurred2, width, height, kernelSize);
        }
    }
    
    cv::Mat grayscale(height, width, CV_8UC1);
    cv::Mat blurred(height, width, CV_8UC1);
    cudaMemcpy(grayscale.data, d_grayscale, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(blurred.data, (numPasses % 2 == 1) ? d_blurred1 : d_blurred2, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    cv::imwrite("output_gray.png", grayscale);
    cv::imwrite("output_blurred.png", blurred);
    
    cudaFree(d_input);
    cudaFree(d_grayscale);
    cudaFree(d_blurred1);
    cudaFree(d_blurred2);
    
    std::cout << "Image processing complete. Outputs saved as 'output_gray.png' and 'output_blurred.png'." << std::endl;
    
    return 0;
}