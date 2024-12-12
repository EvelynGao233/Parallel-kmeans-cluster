#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <sys/stat.h>
#include <string.h> 

#define MAX_ITERATIONS 100

// CUDA kernel to compute distances between points and centroids for arbitrary dimensions
__global__ void get_dst(float* dst, float* points, float* centroids, int n, int k, int dims) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Point index
    int j = threadIdx.y;                          // Centroid index

    if (i < n && j < k) {
        float dist = 0.0f;
        for (int d = 0; d < dims; d++) {
            float diff = points[i * dims + d] - centroids[j * dims + d];
            dist += diff * diff;
        }
        dst[i * k + j] = dist; // Squared Euclidean distance
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
__global__ void clear(float* sum, int* cluster_size, int k, int dims) {
    int j = threadIdx.x;
    if (j < k) {
        for (int d = 0; d < dims; d++) {
            sum[j * dims + d] = 0.0f;
        }
        cluster_size[j] = 0;
    }
}

// CUDA kernel to aggregate points belonging to each cluster
__global__ void recenter_step1(float* sum, int* cluster_size, float* points, int* group, int n, int dims) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int cluster = group[i];
        for (int d = 0; d < dims; d++) {
            atomicAdd(&sum[cluster * dims + d], points[i * dims + d]);
        }
        atomicAdd(&cluster_size[cluster], 1);
    }
}

// CUDA kernel to compute new centroids
__global__ void recenter_step2(float* centroids, float* sum, int* cluster_size, int k, int dims) {
    int j = threadIdx.x;

    if (j < k && cluster_size[j] > 0) {
        for (int d = 0; d < dims; d++) {
            centroids[j * dims + d] = sum[j * dims + d] / cluster_size[j];
        }
    }
}

// K-Means function for variable dimensions
void kmeans(int n, int k, int dims, int max_iters, float* points, float* centroids, int* group) {
    // Device memory
    float *points_d, *centroids_d, *sum_d, *dst_d;
    int *group_d, *cluster_size_d;

    cudaMalloc(&points_d, n * dims * sizeof(float));
    cudaMalloc(&centroids_d, k * dims * sizeof(float));
    cudaMalloc(&dst_d, n * k * sizeof(float));
    cudaMalloc(&group_d, n * sizeof(int));
    cudaMalloc(&sum_d, k * dims * sizeof(float));
    cudaMalloc(&cluster_size_d, k * sizeof(int));

    // Copy data to device
    cudaMemcpy(points_d, points, n * dims * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(centroids_d, centroids, k * dims * sizeof(float), cudaMemcpyHostToDevice);

    // K-Means iterations
    for (int iter = 0; iter < max_iters; iter++) {
        dim3 blockSize(256);
        dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
        get_dst<<<gridSize, blockSize>>>(dst_d, points_d, centroids_d, n, k, dims);
        regroup<<<gridSize, blockSize>>>(group_d, dst_d, n, k);
        clear<<<1, k>>>(sum_d, cluster_size_d, k, dims);
        recenter_step1<<<gridSize, blockSize>>>(sum_d, cluster_size_d, points_d, group_d, n, dims);
        recenter_step2<<<1, k>>>(centroids_d, sum_d, cluster_size_d, k, dims);
    }

    // Copy results back to host
    cudaMemcpy(group, group_d, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(centroids, centroids_d, k * dims * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(points_d);
    cudaFree(centroids_d);
    cudaFree(dst_d);
    cudaFree(group_d);
    cudaFree(sum_d);
    cudaFree(cluster_size_d);
}

// Read points from a file
bool read_points(const std::string& filename, std::vector<float>& points) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    float value;
    while (infile >> value) {
        points.push_back(value);
    }
    infile.close();
    return true;
}

int main() {
    // File names and dataset details
    const char* file_names[] = {
        "DataPoints4D/points_100.txt",
        "DataPoints4D/points_500.txt",
        "DataPoints4D/points_1_000.txt",
        "DataPoints4D/points_10_000.txt",
        "DataPoints4D/points_50_000.txt",
        "DataPoints4D/points_100_000.txt",
        "DataPoints4D/points_250_000.txt",
        "DataPoints4D/points_1_000_000.txt"
    };

    const int num_points[] = {
        100,
        500,
        1000,
        10000,
        50000,
        100000,
        250000,
        1000000
    };

    const int dimensions = 4;  // Specify the dimensions
    const int num_clusters = 10;
    const char* output_folder = "KMeansResults4D";
    mkdir(output_folder, 0777);

    for (int f = 0; f < sizeof(file_names) / sizeof(file_names[0]); f++) {
        std::cout << "Processing file: " << file_names[f] << std::endl;

        std::vector<float> points;
        if (!read_points(file_names[f], points)) {
            std::cerr << "Skipping file due to read error: " << file_names[f] << std::endl;
            continue;
        }

        int n = num_points[f];
        if (points.size() != n * dimensions) {
            std::cerr << "Mismatch between dataset size and dimensions for file: " << file_names[f] << std::endl;
            continue;
        }

        std::vector<float> centroids(num_clusters * dimensions);
        std::vector<int> group(n);

        for (int i = 0; i < num_clusters; i++) {
            for (int d = 0; d < dimensions; d++) {
                centroids[i * dimensions + d] = points[i * dimensions + d];
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        kmeans(n, num_clusters, dimensions, MAX_ITERATIONS, points.data(), centroids.data(), group.data());
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time taken for K-Means on file " << file_names[f] << ": "
                  << elapsed.count() << " seconds" << std::endl;

        const char *base_name = strrchr(file_names[f], '/');
        if (base_name == NULL) {
            base_name = file_names[f]; 
        } else {
            base_name++;
        }

        char output_filename[100];
        sprintf(output_filename, "%s/result_%s", output_folder, base_name);
        std::ofstream output_file(output_filename);
        if (!output_file.is_open()) {
            std::cerr << "Unable to open output file: " << output_filename << std::endl;
            continue;
        }

        output_file << "Final Centroids:\n";
        for (int i = 0; i < num_clusters; i++) {
            output_file << "Centroid " << i + 1 << ": ";
            for (int d = 0; d < dimensions; d++) {
                output_file << centroids[i * dimensions + d] << " ";
            }
            output_file << "\n";
        }
        output_file.close();
    }

    return 0;
}
