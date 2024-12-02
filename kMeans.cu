#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>  

#define MAX_ITERATIONS 100

// Struct for a 3D point
struct Point3D {
    float x, y, z;
};

// CUDA kernel to compute distances between points and centroids
__global__ void get_dst(float* dst, float* x, float* y, float* z, float* mu_x, float* mu_y, float* mu_z, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Point index
    int j = threadIdx.y;                          // Centroid index

    if (i < n && j < k) {
        float dx = x[i] - mu_x[j];
        float dy = y[i] - mu_y[j];
        float dz = z[i] - mu_z[j];
        dst[i * k + j] = dx * dx + dy * dy + dz * dz; // Squared Euclidean distance
    }
}

// CUDA kernel to assign points to the nearest cluster
__global__ void regroup(int* group, float* dst, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float min_dst = dst[i * k];
        int best_cluster = 0;

        for (int j = 1; j < k; j++) {
            float d = dst[i * k + j];
            if (d < min_dst) {
                min_dst = d;
                best_cluster = j;
            }
        }
        group[i] = best_cluster;
    }
}

// CUDA kernel to clear the cluster sum accumulators
__global__ void clear(float* sum_x, float* sum_y, float* sum_z, int* cluster_size, int k) {
    int j = threadIdx.x;
    if (j < k) {
        sum_x[j] = 0.0f;
        sum_y[j] = 0.0f;
        sum_z[j] = 0.0f;
        cluster_size[j] = 0;
    }
}

// CUDA kernel to aggregate points belonging to each cluster
__global__ void recenter_step1(float* sum_x, float* sum_y, float* sum_z, int* cluster_size, float* x, float* y, float* z, int* group, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int cluster = group[i];
        atomicAdd(&sum_x[cluster], x[i]);
        atomicAdd(&sum_y[cluster], y[i]);
        atomicAdd(&sum_z[cluster], z[i]);
        atomicAdd(&cluster_size[cluster], 1);
    }
}

// CUDA kernel to compute new centroids
__global__ void recenter_step2(float* mu_x, float* mu_y, float* mu_z, float* sum_x, float* sum_y, float* sum_z, int* cluster_size, int k) {
    int j = threadIdx.x;

    if (j < k && cluster_size[j] > 0) {
        mu_x[j] = sum_x[j] / cluster_size[j];
        mu_y[j] = sum_y[j] / cluster_size[j];
        mu_z[j] = sum_z[j] / cluster_size[j];
    }
}

// K-Means function
void kmeans(int n, int k, int max_iters, float* x, float* y, float* z, float* mu_x, float* mu_y, float* mu_z, int* group) {
    // Device memory
    float *x_d, *y_d, *z_d, *mu_x_d, *mu_y_d, *mu_z_d, *sum_x_d, *sum_y_d, *sum_z_d, *dst_d;
    int *group_d, *cluster_size_d;

    cudaMalloc(&x_d, n * sizeof(float));
    cudaMalloc(&y_d, n * sizeof(float));
    cudaMalloc(&z_d, n * sizeof(float));
    cudaMalloc(&mu_x_d, k * sizeof(float));
    cudaMalloc(&mu_y_d, k * sizeof(float));
    cudaMalloc(&mu_z_d, k * sizeof(float));
    cudaMalloc(&dst_d, n * k * sizeof(float));
    cudaMalloc(&group_d, n * sizeof(int));
    cudaMalloc(&sum_x_d, k * sizeof(float));
    cudaMalloc(&sum_y_d, k * sizeof(float));
    cudaMalloc(&sum_z_d, k * sizeof(float));
    cudaMalloc(&cluster_size_d, k * sizeof(int));

    // Copy data to device
    cudaMemcpy(x_d, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(z_d, z, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mu_x_d, mu_x, k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mu_y_d, mu_y, k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mu_z_d, mu_z, k * sizeof(float), cudaMemcpyHostToDevice);

    // K-Means iterations
    for (int iter = 0; iter < max_iters; iter++) {
        // Launch kernels
        dim3 blockSize(256);
        dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
        auto start = std::chrono::high_resolution_clock::now();
        get_dst<<<gridSize, blockSize>>>(dst_d, x_d, y_d, z_d, mu_x_d, mu_y_d, mu_z_d, n, k);
        regroup<<<gridSize, blockSize>>>(group_d, dst_d, n, k);
        clear<<<1, k>>>(sum_x_d, sum_y_d, sum_z_d, cluster_size_d, k);
        recenter_step1<<<gridSize, blockSize>>>(sum_x_d, sum_y_d, sum_z_d, cluster_size_d, x_d, y_d, z_d, group_d, n);
        recenter_step2<<<1, k>>>(mu_x_d, mu_y_d, mu_z_d, sum_x_d, sum_y_d, sum_z_d, cluster_size_d, k);
        
    }

    // Copy results back to host
    cudaMemcpy(group, group_d, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(mu_x, mu_x_d, k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(mu_y, mu_y_d, k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(mu_z, mu_z_d, k * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
    cudaFree(mu_x_d);
    cudaFree(mu_y_d);
    cudaFree(mu_z_d);
    cudaFree(dst_d);
    cudaFree(group_d);
    cudaFree(sum_x_d);
    cudaFree(sum_y_d);
    cudaFree(sum_z_d);
    cudaFree(cluster_size_d);
}

// Read points from a file
bool read_points(const std::string& filename, std::vector<Point3D>& points) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    float x, y, z;
    while (infile >> x >> y >> z) {
        points.push_back({x, y, z});
    }
    infile.close();
    return true;
}

int main() {
    const int k = 10; // Number of clusters
    const int max_iters = MAX_ITERATIONS;

    // Array of filenames to process
    std::vector<std::string> filenames = {
        "datapoints/points_100.txt",
        "datapoints/points_500.txt",
        "datapoints/points_1_000.txt",
        "datapoints/points_10_000.txt",
        "datapoints/points_50_000.txt",
        "datapoints/points_100_000.txt",
        "datapoints/points_250_000.txt",
        "datapoints/points_1_000_000.txt"
    };
    // Loop over each file
    for (const auto& filename : filenames) {
        std::cout << "Processing file: " << filename << std::endl;
        // Read points from the file
        std::vector<Point3D> points;
        if (!read_points(filename, points)) {
            std::cerr << "Skipping file due to read error: " << filename << std::endl;
            continue;
        }
        int n = points.size();
        std::vector<float> x(n), y(n), z(n);
        for (int i = 0; i < n; i++) {
            x[i] = points[i].x;
            y[i] = points[i].y;
            z[i] = points[i].z;
        }
        std::vector<float> mu_x(k), mu_y(k), mu_z(k);
        std::vector<int> group(n);

        // Initialize centroids (use the first k points as initial centroids)
        for (int j = 0; j < k; j++) {
            mu_x[j] = points[j].x;
            mu_y[j] = points[j].y;
            mu_z[j] = points[j].z;
        }

        // Run K-Means and measure runtime
        auto start = std::chrono::high_resolution_clock::now();
        kmeans(n, k, max_iters, x.data(), y.data(), z.data(), mu_x.data(), mu_y.data(), mu_z.data(), group.data());
        auto end = std::chrono::high_resolution_clock::now();

        // Print results
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time taken for K-Means on file " << filename << ": " 
                  << elapsed.count() << " seconds" << std::endl;
        std::cout << "-------------------------------------------\n";
    }

    return 0;
}
