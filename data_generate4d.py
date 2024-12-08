import os
import numpy as np

def generate_4d_points(num_points, filename, folder):
    cluster_centers_and_spreads = [
        ([-13.5, 89.0, 10.0, 20.0], 5),
        ([-8.3, 80.7, 20.0, -15.0], 10),
        ([-19.8, 82.1, -15.0, 5.0], 15),
        ([-26.2, 87.9, 5.0, 12.0], 7),
        ([-23.2, 88.4, 0.0, -8.0], 12),
        ([86.3, -86.0, -20.0, 30.0], 8),
        ([-65.9, -36.2, 30.0, 25.0], 20),
        ([81.6, 54.5, -25.0, -10.0], 6),
        ([34.8, 96.9, 15.0, 18.0], 10),
        ([-28.4, 83.9, -10.0, -5.0], 5),
    ]

    points = []
    for center, spread in cluster_centers_and_spreads:
        cluster_points = np.random.normal(
            loc=center, 
            scale=spread, 
            size=(num_points // len(cluster_centers_and_spreads), 4)
        )
        points.append(cluster_points)
    points = np.vstack(points)
    
    filepath = os.path.join(folder, filename)
    np.savetxt(filepath, points, fmt="%.8f")

def main():
    folder = "DataPoints4D"
    os.makedirs(folder, exist_ok=True)

    datasets = [
        (100, "points_100.txt"),
        (500, "points_500.txt"),
        (1000, "points_1_000.txt"),
        (10000, "points_10_000.txt"),
        (50000, "points_50_000.txt"),
        (100000, "points_100_000.txt"),
        (250000, "points_250_000.txt"),
        (1000000, "points_1_000_000.txt"),
    ]

    for num_points, filename in datasets:
        generate_4d_points(num_points, filename, folder)

if __name__ == "__main__":
    main()
