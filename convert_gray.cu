#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

__global__ void rgb_to_grayscale(const uchar3* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        uchar3 rgb = input[idx];
        output[idx] = static_cast<unsigned char>(0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z);
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
    unsigned char* d_output;
    cudaMalloc(&d_input, width * height * sizeof(uchar3));
    cudaMalloc(&d_output, width * height * sizeof(unsigned char));

    cudaMemcpy(d_input, input.data, width * height * sizeof(uchar3), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    rgb_to_grayscale<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cv::Mat output(height, width, CV_8UC1);
    cudaMemcpy(output.data, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::imwrite("output_gray.jpg", output);

    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "Grayscale conversion complete. Output saved as 'output_gray.jpg'." << std::endl;

    return 0;
}