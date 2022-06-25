import os 
import numpy as np 
import cv2 
from typing import Tuple, List  


class VisualOdometry:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        # load camera intrinsic matrix K and projection matrix P  
        self.K, self.P = self._load_calib(
            os.path.join(self.data_dir, 'calib.txt')
        )
        self.gt_poses = self._load_poses(
            os.path.join(self.data_dir, 'poses.txt')
        )
        self.images = self._load_images(
            os.path.join(self.data_dir, 'image_l')
        )
        # load orb algorithm and limit features to max 3000 
        self.feature_detector = cv2.ORB_create(nfeatures=3000)
        # use the Flann matcher  
        self.matcher = cv2.FlannBasedMatcher(
            indexParams={
                'algorithm': 6, # flann index lsh 
                'table_number': 6,
                'key_size': 12,
                'multi_probe_level': 1 
            }, 
            searchParams={
                'checks': 50
            }
        ) 


    @staticmethod
    def _load_calib(path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load camera calibration parameters.

        Args:
            path (str): File path to calibration.txt. 

        Returns:
            Tuple[np.ndarray, np.ndarray]: Intrinsic parameters K, 
                                           Projection matrix P 
        """
        with open(path, 'r') as fp: 
            parameters = np.fromstring(
                fp.readline(), dtype=np.float64, sep=' ',
            )
            P = np.reshape(parameters, (3,4))
            K = P[0:3, 0:3] 
        
        return K, P


    @staticmethod
    def _load_poses(path: str) -> List[np.ndarray]:
        
        poses = []
        with open(path, 'r') as fp: 
            for line in fp.readlines():
                pose = np.fromstring(line, dtype=np.float64, sep=' ')
                pose = pose.reshape(3,4)
                pose = np.vstack((pose, [0, 0, 0, 1]))
                poses.append(pose)

        return poses 


    @staticmethod 
    def _load_images(path: str) -> List[np.ndarray]:
        """""" 
        image_paths = [
            os.path.join(path, filename) for filename in sorted(os.listdir(path))
        ]
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

        return images


    @staticmethod 
    def _form_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Create Transformation matrix from the rotation matrix 
        and translation vector

        Args:
            R (np.ndarray): rotation matrix  
            t (np.ndarray): translation vector  

        Returns:
            np.ndarray: transformation matrix 
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R 
        T[:3, 3] = t 

        return T 

    
    def get_matches(self, idx: int, visual: bool = False): 
        
        kpts1, desc1 = self.feature_detector.detectAndCompute(
            self.images[idx - 1], None,
        ) 
        kpts2, desc2 = self.feature_detector.detectAndCompute(
            self.images[idx], None,
        )

        matches = self.matcher.knnMatch(
            desc1, desc2, k=2,
        )

        # store all good matches, use per Lowe's ratio 
        goods = []
        for m,n in matches: 
            if m.distance < 0.5 * n.distance: 
                goods.append(m)

        points1 = np.float32([
            kpts1[m.queryIdx].pt for m in goods
        ])
        points2 = np.float32([
            kpts2[m.trainIdx].pt for m in goods
        ])

        if visual:
            draw_params = {
                'matchColor': -1,
                'singlePointColor': None, 
                'matchesMask': None,
                'flags': 2,
            }

            img = cv2.drawMatches(
                self.images[idx],
                kpts1, 
                self.images[idx - 1],
                kpts2,
                goods,
                None,
                **draw_params,
            )

            cv2.imshow("Good feature matches", img)
            cv2.waitKey(750)
        
        return points1, points2 


    def get_pose(self, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        
        essential_matrix, _= cv2.findEssentialMat(
            points1,
            points2,
            self.K,
        ) 
        R, t = self.decompose_ess_mat(essential_matrix, points1, points2)
        transformation_matrix = self._form_transform(R, t) 

        return transformation_matrix 


    def decompose_ess_mat(
        self, 
        essential_matrix: np.ndarray, 
        points1: np.ndarray, 
        points2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        R1, R2, t = cv2.decomposeEssentialMat(essential_matrix)
        T1 = self._form_transform(R1, np.ndarray.flatten(t))
        T2 = self._form_transform(R2, np.ndarray.flatten(t))
        T3 = self._form_transform(R1, np.ndarray.flatten(-t))
        T4 = self._form_transform(R2, np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]

        # homogenize K 
        K = np.concatenate((self.K, np.zeros((3, 1))), axis=1)

        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        positives = []

        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(
                self.P, P, points1.T, points2.T)
            hom_Q2 = T @ hom_Q1

            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0) 
            relative_scale = np.mean(
                np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1) / 
                np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1)
            )
            positives.append(total_sum + relative_scale)

        # find the correct solution
        max_ = np.argmax(positives)
        if (max_ == 0):
            return R1, np.ndarray.flatten(t)
        elif (max_ == 1):
            return R2, np.ndarray.flatten(t)
        elif (max_ == 2):
            return R1, np.ndarray.flatten(-t)
        elif (max_ == 3):
            return R2, np.ndarray.flatten(-t)