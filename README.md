# Parallel-kmeans-cluster
Overview
This project implements the K-means clustering algorithm in sequential and parallelized versions using OpenMP and CUDA. K-means is a widely used clustering technique in machine learning and data analysis for partitioning data points into distinct groups based on their distances to centroids.

The repository includes:

Sequential implementation in C
OpenMP implementation for parallelization on CPUs
CUDA implementation for parallelization on GPUs
Tools for generating 3D and 4D data points
Features
Sequential K-means
A basic implementation of the K-means clustering algorithm in C.

Parallel K-means with OpenMP
This version uses OpenMP to parallelize the K-means algorithm on multi-core CPUs for improved performance.

Parallel K-means with CUDA
This version leverages CUDA to parallelize the algorithm on NVIDIA GPUs, achieving significant speedups for large datasets.

Data Generation Tools
Python scripts for generating synthetic 3D and 4D data points to test the K-means algorithms.
