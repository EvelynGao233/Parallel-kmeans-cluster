#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define NUM_THREADS 8
#define MAX_ITERATIONS 1000

typedef struct {
    double *coords;
} Point;

double euclidean_distance(Point a, Point b, int dimensions) {
    double dist = 0.0;
    for (int i = 0; i < dimensions; i++) {
        dist += pow(a.coords[i] - b.coords[i], 2);
    }
    return sqrt(dist);
}

void read_points_from_file(const char *filename, int num_points, int dimensions, Point *points) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Unable to open file %s.\n", filename);
        exit(1);
    }

    for (int i = 0; i < num_points; i++) {
        points[i].coords = (double *)malloc(dimensions * sizeof(double));
        for (int j = 0; j < dimensions; j++) {
            fscanf(file, "%lf", &points[i].coords[j]);
        }
    }
    fclose(file);
}

void normalize_points(Point *points, int num_points, int dimensions) {
    double *min_vals = (double *)malloc(dimensions * sizeof(double));
    double *max_vals = (double *)malloc(dimensions * sizeof(double));

    for (int i = 0; i < dimensions; i++) {
        min_vals[i] = INFINITY;
        max_vals[i] = -INFINITY;
    }

    #pragma omp parallel for
    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < dimensions; j++) {
            #pragma omp critical
            {
                if (points[i].coords[j] < min_vals[j]) min_vals[j] = points[i].coords[j];
                if (points[i].coords[j] > max_vals[j]) max_vals[j] = points[i].coords[j];
            }
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < dimensions; j++) {
            points[i].coords[j] = (points[i].coords[j] - min_vals[j]) / (max_vals[j] - min_vals[j]);
        }
    }

    free(min_vals);
    free(max_vals);
}

void k_means_clustering(const char *filename, int num_points, Point *points, int num_clusters, int dimensions, const char *output_folder) {
    Point centroids[num_clusters];

    for (int i = 0; i < num_clusters; i++) {
        centroids[i].coords = (double *)malloc(dimensions * sizeof(double));
        memcpy(centroids[i].coords, points[i].coords, dimensions * sizeof(double));
    }

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        int cluster_counts[num_clusters];
        Point sum[num_clusters];
        int converged = 1;

        memset(cluster_counts, 0, sizeof(cluster_counts));
        for (int i = 0; i < num_clusters; i++) {
            sum[i].coords = (double *)calloc(dimensions, sizeof(double));
        }

        #pragma omp parallel for
        for (int i = 0; i < num_points; i++) {
            double min_dist = INFINITY;
            int closest_centroid = -1;

            for (int j = 0; j < num_clusters; j++) {
                double dist = euclidean_distance(points[i], centroids[j], dimensions);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_centroid = j;
                }
            }

            if (closest_centroid != -1) {
                #pragma omp critical
                {
                    cluster_counts[closest_centroid]++;
                    for (int d = 0; d < dimensions; d++) {
                        sum[closest_centroid].coords[d] += points[i].coords[d];
                    }
                }
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < num_clusters; i++) {
            if (cluster_counts[i] > 0) {
                for (int d = 0; d < dimensions; d++) {
                    double new_value = sum[i].coords[d] / cluster_counts[i];
                    if (fabs(centroids[i].coords[d] - new_value) > 0.0001) {
                        centroids[i].coords[d] = new_value;
                        converged = 0;
                    }
                }
            }
        }

        for (int i = 0; i < num_clusters; i++) {
            free(sum[i].coords);
        }

        if (converged) {
            break;
        }
    }

    char output_filename[256];
    snprintf(output_filename, sizeof(output_filename), "%s/result_%s.txt", output_folder, strrchr(filename, '/') + 1);
    FILE *output_file = fopen(output_filename, "w");
    if (!output_file) {
        fprintf(stderr, "Unable to open output file %s.\n", output_filename);
        exit(1);
    }

    fprintf(output_file, "Final Centroids for %s:\n", filename);
    for (int i = 0; i < num_clusters; i++) {
        fprintf(output_file, "Centroid %d: ", i + 1);
        for (int d = 0; d < dimensions; d++) {
            fprintf(output_file, "%.4lf ", centroids[i].coords[d]);
        }
        fprintf(output_file, "\n");
        free(centroids[i].coords);
    }
    fclose(output_file);
}

int main() {
    omp_set_num_threads(NUM_THREADS);

    const char *file_names[] = {
        "points_100.txt",
        "points_500.txt",
        "points_1_000.txt",
        "points_10_000.txt",
        "points_50_000.txt",
        "points_100_000.txt",
        "points_250_000.txt",
        "points_1_000_000.txt"
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

    const char *output_folder = "OpenMPKMeansResults3D";
    mkdir(output_folder, 0777);

    const int dimensions = 3;
    const int num_clusters = 10;

    for (int f = 0; f < sizeof(file_names) / sizeof(file_names[0]); f++) {
        char filename[50];
        snprintf(filename, sizeof(filename), "DataPoints3D/%s", file_names[f]);

        int num = num_points[f];
        Point *points = (Point *)malloc(num * sizeof(Point));
        if (!points) {
            fprintf(stderr, "Memory allocation failed.\n");
            return 1;
        }

        read_points_from_file(filename, num, dimensions, points);
        normalize_points(points, num, dimensions);

        double start_time = omp_get_wtime();
        k_means_clustering(filename, num, points, num_clusters, dimensions, output_folder);
        double end_time = omp_get_wtime();

        printf("Execution Time for %s: %f seconds\n", filename, end_time - start_time);

        for (int i = 0; i < num; i++) {
            free(points[i].coords);
        }
        free(points);
    }

    return 0;
}
