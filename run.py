import numpy as np 
from tqdm import tqdm 

from monocular import VisualOdometry
from utils import plot_results 


def main():
    data_dir = "data/KITTI_sequence_1"
    visual_odometry = VisualOdometry(data_dir=data_dir)

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

            print(f"\nGround truth pose: {str(gt_pose)}")
            print(f"\nCurrent pose: {str(cur_pose)}")
            print("--" * 25)
            print(f"Current pose x,y: {str(cur_pose[0, 3])}  {str(cur_pose[2, 3])}")
            print(f"GT pose x,y: {str(gt_pose[0, 3])}  {str(gt_pose[2, 3])}")

        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

    plot_results(gt_path, estimated_path)


if __name__ == "__main__":
    main()

