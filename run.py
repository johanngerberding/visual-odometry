from xml.sax.handler import feature_validation
import numpy as np 
from tqdm import tqdm 

from monocular import VisualOdometry, VALID_FEATURE_DETECTORS
from utils import plot_results 


def estimate_path(data_dir: str, feature_detector: str, verbose: bool = False):
    
    visual_odometry = VisualOdometry(data_dir=data_dir, feature_detector=feature_detector)
    gt_path = []
    estimated_path = []

    for i, gt_pose in enumerate(tqdm(visual_odometry.gt_poses, unit="pose")):
        if i == 0: 
            cur_pose = gt_pose 
        else: 
            points1, points2 = visual_odometry.get_matches(i, visual=False)
            transformation = visual_odometry.get_pose(
                points1, points2,
            )
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transformation))

            if verbose:
                print(f"\nGround truth pose: {str(gt_pose)}")
                print(f"\nCurrent pose: {str(cur_pose)}")
                print("--" * 25)
                print(f"Current pose x,z: {str(cur_pose[0, 3])}  {str(cur_pose[2, 3])}")
                print(f"GT pose x,z: {str(gt_pose[0, 3])}  {str(gt_pose[2, 3])}")

        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

    return gt_path, estimated_path


def main():
    data_dir = "data/KITTI_sequence_1"
    verbose = False

    for feature_detector in VALID_FEATURE_DETECTORS:
        gt_path, estimated_path = estimate_path(
            data_dir=data_dir, 
            feature_detector=feature_detector, 
            verbose=verbose,
        )

        plot_results(gt_path, estimated_path, f"Feature detector: {feature_detector}")


if __name__ == "__main__":
    main()

