"""import numpy as np
import cv2

translation_range = 20  # Maximum translation in pixels
rotation_range_per_video = 30  # Maximum rotation angle per video in degrees
scale_range = 0.2  # Maximum scaling factor
noise_stddev = 2.0  # Standard deviation for Gaussian noise


# Load your skeleton keypoints data (assuming it's a NumPy array of shape [num_samples, num_skeletons, num_keypoints, 2])
# Replace "skeleton_keypoints_data" with your actual data
pose_keypoints_list = np.load("skeletons.npy", allow_pickle=True)


print(pose_keypoints_list)
# Convert pose keypoints to Euclidean vectors (without prob. not euclidean but too lazy too change)
euclidean_pose_keypoints_list = []

for video_pose_keypoints in pose_keypoints_list:
    euclidean_pose_keypoints = video_pose_keypoints[:, :, :2]
    euclidean_pose_keypoints_list.append(euclidean_pose_keypoints)

skeleton_keypoints_data = euclidean_pose_keypoints_list



# Create an empty array to store the augmented keypoints
augmented_keypoints = []

# Perform data augmentation on each video
for video in skeleton_keypoints_data:
    rotation_angle = np.random.uniform(-rotation_range_per_video, rotation_range_per_video)

    augmented_frames = []
    for skeleton in video:
        augmented_skeleton = []

        #rotation
        for keypoint in skeleton:
            # Rotate the keypoint
            rotated_keypoint = cv2.rotate(keypoint, cv2.ROTATE_90_CLOCKWISE)

            # Append the rotated keypoint to the augmented skeleton
            augmented_skeleton.append(rotated_keypoint)

        # Convert the augmented skeleton to a NumPy array
        augmented_skeleton = np.array(augmented_skeleton)

        # Append the augmented skeleton to the augmented frames
        augmented_frames.append(augmented_skeleton)

    # Convert the augmented frames to a NumPy array
    augmented_frames = np.array(augmented_frames)

    # Append the augmented frames to the augmented keypoints
    augmented_keypoints.append(augmented_frames)

# Convert the augmented keypoints to a NumPy array
augmented_keypoints = np.array(augmented_keypoints)

print(augmented_keypoints.shape)
# Save the augmented keypoints
# Replace "augmented_skeleton_keypoints.npy" with your desired file name
np.save("augmented_skeleton_keypoints.npy", augmented_keypoints)"""
import numpy as np
import cv2
# Load your skeleton keypoints data (assuming it's a NumPy array of shape [num_samples, num_skeletons, num_keypoints, 2])
# Replace "skeleton_keypoints_data" with your actual data


# Load the original videos
videos = np.load("/home/nuno/Documents/universiteit/tweedejaarsproject/real/extra2Dskeletons/skeleton_data_files.npy", allow_pickle=True)

# Load the original labels
labels = np.load("labels_posenet_files.npy", allow_pickle=True)

# Define the number of augmentations per video
num_augmentations = 50

# Create a new label list for the augmented videos
augmented_labels = np.repeat(labels, num_augmentations, axis=0)

# Save the augmented labels as a numpy array
np.save('augmented_labels.npy', augmented_labels)


import numpy as np

# Define augmentation parameters
num_augmentations = 50  # Number of augmentations per video
translation_range = 0.05  # Maximum translation distance in pixels
rotation_range = 5  # Maximum rotation angle in degrees

# Get the shape of the original videos array
original_shape = videos.shape

# Calculate the mean of each skeleton in the array
mean_skeleton = np.mean(videos, axis=(2, 3, 4))

# Reshape the videos array to a 2D shape
videos_2d = videos.reshape((original_shape[0], -1))

# Initialize the augmented array
augmented_videos = np.zeros((original_shape[0] * num_augmentations, *original_shape[1:]))

# Iterate over each video
for video_idx in range(original_shape[0]):
    # Get the current video and its mean skeleton
    video = videos_2d[video_idx]
    mean_skeleton = np.mean(video, axis=0)

    # Repeat the video for each augmentation
    repeated_video = np.repeat(video[np.newaxis, :], num_augmentations, axis=0)

    # Augment each repeated video
    for aug_idx in range(num_augmentations):
        # Get the current repeated video
        repeated_video_aug = repeated_video[aug_idx]

        # Reshape the repeated video to the original shape
        augmented_skeletons = repeated_video_aug.reshape(original_shape[1:])

        # Augment each skeleton in the video
        for skeleton_idx in range(augmented_skeletons.shape[0]):
            # Translate the skeleton randomly around the mean
            translation = np.random.uniform(low=-translation_range, high=translation_range, size=(2, 1))
            translated_skeleton = augmented_skeletons[skeleton_idx] + translation

            # Rotate the skeleton randomly around the mean
            rotation_angle = np.random.uniform(low=-rotation_range, high=rotation_range)
            rotation_matrix = np.array([[np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle))],
                                        [np.sin(np.radians(rotation_angle)), np.cos(np.radians(rotation_angle))]])
            rotated_skeleton = np.dot(rotation_matrix, translated_skeleton - mean_skeleton[skeleton_idx]) + mean_skeleton[skeleton_idx]

            # Store the augmented skeleton
            augmented_skeletons[skeleton_idx] = rotated_skeleton

        # Store the augmented video
        augmented_videos[video_idx * num_augmentations + aug_idx] = augmented_skeletons

# Save the augmented videos as a numpy array
np.save('augmented_videos.npy', augmented_videos)



# Create empty arrays to store the augmented keypoints and labels
"""augmented_keypoints = []
augmented_labels = []

# Perform data augmentation on each video
for video, label in zip(skeleton_keypoints_data, labels):
    for _ in range(10):  # Generate 10 augmented videos per original video
        rotation_angle = np.random.uniform(-rotation_range_per_video, rotation_range_per_video)
        scale_factor = np.random.uniform(1 - scale_range, 1 + scale_range)

        augmented_frames = []
        for skeleton in video:
            augmented_skeleton = []

            # Apply augmentation operations; rotate, scale, translate, and add noise but keep the same number of keypoints; 17
            for keypoint in skeleton:
                # translate the keypoint
                translated_keypoint = keypoint + np.random.uniform(-translation_range, translation_range, size=2)
                # Append the noised keypoint to the augmented skeleton
                augmented_skeleton.append(translated_keypoint)
            # Append the scaled skeleton to the augmented frames
            #now rotate the skeleton around the midpoint of the skeleton
            #first find the midpoint
            midpoint = np.mean(augmented_skeleton, axis=0)
            print(midpoint)
            #now rotate the skeleton around the midpoint with the rotation angle
            rotated_skeleton = []
            for keypoint in augmented_skeleton:
                # translate the keypoint
                translated_keypoint = keypoint - midpoint
                print(translated_keypoint)
                # rotate using the rotation matrix and the rotation angle
                rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],[np.sin(rotation_angle), np.cos(rotation_angle)]])
                print(rotation_matrix)
                # rotate the keypoint using the matrix and dot product
                rotated_keypoint = np.dot(rotation_matrix, translated_keypoint)
                print(rotated_keypoint)
                # translate the keypoint back to the original position
                translated_keypoint = rotated_keypoint + midpoint
                # Append the rotated keypoint to the augmented skeleton
                rotated_skeleton.append(translated_keypoint)

            augmented_frames.append(rotated_skeleton)

        # Convert the augmented frames to a NumPy array
        augmented_frames = np.array(augmented_frames)

        # Ensure all frames have the same shape
        frame_shape = augmented_frames[0].shape
        augmented_frames = np.array([cv2.resize(frame, frame_shape[::-1]) for frame in augmented_frames])

        # Pad or truncate keypoints to a fixed number (17 in this case)
        num_keypoints = augmented_frames.shape[0]
        if num_keypoints < 17:
            pad_amount = 17 - num_keypoints
            padded_frames = np.pad(augmented_frames, [(0, pad_amount), (0, 0)], mode='constant')
            augmented_frames = padded_frames
        elif num_keypoints > 17:
            augmented_frames = augmented_frames[:17]

        # Append the augmented frames to the augmented keypoints
        augmented_keypoints.append(augmented_frames)

        # Append the label corresponding to the augmented video
        augmented_labels.append(label)

# Convert the augmented keypoints and labels to NumPy arrays
augmented_keypoints = np.array(augmented_keypoints)
augmented_labels = np.array(augmented_labels)


# Save the augmented keypoints and labels
# Replace "augmented_skeleton_keypoints.npy" and "augmented_labels.npy" with your desired file names
np.save("augmented_skeleton_keypoints.npy", augmented_keypoints)
np.save("augmented_labels.npy", augmented_labels)


"""

