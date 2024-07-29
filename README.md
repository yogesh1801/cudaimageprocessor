# CUDA Image Processing: Grayscale Conversion and Blur

This project demonstrates basic image processing techniques using CUDA and OpenCV. It includes functionality for converting a color image to grayscale and applying a customizable box blur effect.

## Features

- RGB to Grayscale conversion using CUDA
- Customizable Box Blur implementation with CUDA
- Multiple blur passes for stronger effect
- Efficient use of GPU memory with alternating buffers

## Prerequisites

- CUDA Toolkit (version 10.0 or later recommended)
- OpenCV (version 4.0 or later recommended)
- A CUDA-capable GPU
- C++ compiler with C++11 support

## Building the Project

1. Ensure you have the CUDA Toolkit and OpenCV installed on your system.

2. Clone this repository:
```sh
https://github.com/yogesh1801/cudaimageprocessor.git
```

3.Compile the project:
```sh
nvcc -std=c++11 main.cu -o image_processor pkg-config --cflags --libs opencv4

# You may use cmake files to directly configure the proect.
```

## Usage
1. Place your input image in the same directory as the executable and name it `img.png`.
2. Run the program: