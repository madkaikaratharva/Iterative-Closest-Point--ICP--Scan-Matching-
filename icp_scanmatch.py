import numpy as np
from utils import mean_dist
import open3d as o3d
import datetime
from sklearn.neighbors import NearestNeighbors

class IcpScanMatch():
    """
    Class for performing Iterative Closest Point (ICP) scan matching on 3D point clouds.
    """

    def __init__(self, points1, points2,
                 n_iters=100,
                 threshold=0.01,
                 correspondenc=False):
        
        # Initializtion
        self.points1 = points1
        self.points2 = points2
        self.n_iters = n_iters
        self.threshold = threshold
        self.correspondence =  correspondenc

    
    def icp_core(self, points1, points2):
        """
        Solves transformation from points1 to points2

        params:
        - points1 (numpy.ndarray): The first point cloud represented as a numpy array of shape (n, 3), where n is the number of points.
        - points2 (numpy.ndarray): The second point cloud represented as a numpy array of shape (n, 3), where n is the number of points.

        returns:
        - Transformation matrix T
        """

        # Checks if the number of points matches
        assert points1.shape == points2.shape, f"Point cloud shapes do not match: {points1.shape} vs {points2.shape}"

        # Initialize Transformation matrix
        T = np.eye(4)

        # Calculate Centroid
        centroid_1 = np.mean(points1, axis=0)
        centroid_1 = centroid_1.reshape((1, 3))

        centroid_2 = np.mean(points2, axis=0) 
        centroid_2 = centroid_2.reshape((1, 3))

        # Center points around the mean
        points1 = points1 - centroid_1
        points2 = points2 - centroid_2

        # Coumpute H
        H = np.dot(points1.T, points2)

        # Perform SVD on H matrix to solve for R and t
        U, D, V_transpose = np.linalg.svd(H)
        
        V = V_transpose.T

        # Evaluate R and t
        R = np.dot(V, U.T)

        t = centroid_2.T - np.dot(R, centroid_1.T)

        # Form the transformation matrix T
        T[0:3, 0:3] = R
        T[0:3, 3:4] = t

        return T
    
    def scan_matching(self):
        """
        Performs ICP scan matching and visualize the results.
        """

        if self.correspondence:
            print("Known Correspondence")

            # Get transformation matrix T
            T = self.icp_core(self.points1, self.points2)

            print("-------Transformation Matrix-------")
            print(T)

            # Get the transformed point cloud using T
            self.points1_transformed = np.dot(T[0:3, 0:3], self.points1.T) + T[0:3, 3:4]
            self.points1_transformed = self.points1_transformed.T

            # Evaluate error between the target frame and transformed frame 
            error = mean_dist(self.points1_transformed, self.points2)
            print('mean error: ' + str(error))

            # Visulalization using open3d
            axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(self.points1)
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(self.points2)
            pcd1_transformed = o3d.geometry.PointCloud()
            pcd1_transformed.points = o3d.utility.Vector3dVector(self.points1_transformed)
            pcd1.paint_uniform_color([1, 0, 0])
            pcd2.paint_uniform_color([0, 1, 0])
            pcd1_transformed.paint_uniform_color([0, 0, 1])
            o3d.visualization.draw_geometries([pcd1, pcd2, pcd1_transformed, axis_pcd])
        else:
            print("Unknown Correspondence")

            points1 = self.points1.copy()
            points2 = self.points2.copy()

            # Initialize Accumulated T
            T_accumulated = np.eye(4)

            # Enable Open3d visualizer
            axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(points1)
            pcd1.paint_uniform_color([1, 0, 0])
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(points2)
            pcd2.paint_uniform_color([0, 1, 0])
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(axis_pcd)
            vis.add_geometry(pcd1)
            vis.add_geometry(pcd2)

            start_time = datetime.datetime.now()

            for i in range(self.n_iters):
         
                # Generate points2_nearest. This will the target frame
                points2_nearest = self.points1.copy()

                # For all points in points1, find the nearest points in points2. Assign it to points2_nearest
                nearest_points = NearestNeighbors(n_neighbors=1)
                nearest_points.fit(points2)

                indices = nearest_points.kneighbors(points1, return_distance=False)

                # Update points2_nearest based on the indices obtained
                index_nearest = 0

                for index in indices:
                    points2_nearest[index_nearest] = points2[index]
                    index_nearest += 1

                # Get the transformation matrix T
                T = self.icp_core(points1, points2_nearest)

                # Update T_accumulated 
                T_accumulated = np.dot(T, T_accumulated)

                # Transform and Update points1
                points1 = np.dot(T[0:3, 0:3], points1.T) + T[0:3, 3:4] 
                points1 = points1.T

                # Evaluate error between the target frame and transformed frame
                error = mean_dist(points1, points2) 
                print('mean error: ' + str(error))

                # Add transformed points to visualizer
                pcd1_transformed = o3d.geometry.PointCloud()
                pcd1_transformed.points = o3d.utility.Vector3dVector(points1)
                pcd1_transformed.paint_uniform_color([0, 0, 1])
                vis.add_geometry(pcd1_transformed)
                vis.poll_events()
                vis.update_renderer()

                vis.remove_geometry(pcd1_transformed)

                if error < 0.00001 or error < self.threshold:
                    print('fully converged!')
                    break
            
            end_time = datetime.datetime.now()
            time_difference = (end_time - start_time).total_seconds()
            print('time cost: ' + str(time_difference) + ' s')
            vis.destroy_window()

            o3d.visualization.draw_geometries([axis_pcd, pcd1, pcd1_transformed, pcd2])
