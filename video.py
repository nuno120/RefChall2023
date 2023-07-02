import argparse
import cv2
import numpy as np
import torch
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width

class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


import cv2
import numpy as np
import torch
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width
import time

import cv2
import numpy as np
import time

import time

def process_video(video_path):
    net = PoseEstimationWithMobileNet()
    checkpoint_path = '/home/nuno/Documents/GitHub/lightweight-human-pose-estimation.pytorch/checkpoint_iter_370000.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    frames_per_second = 2  # Number of frames per second to process
    minimum_video_length = 12  # Minimum length of the video in seconds
    # Process each video in the dataset
    count = 0
    cap = cv2.VideoCapture(video_path)
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        print('continue')
        return None, None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length = frame_count / video_fps
    # Calculate the number of frames to process based on the desired frames per second and minimum video length
    total_frames_video = frames_per_second * minimum_video_length 

    # Calculate the frame skip value based on the desired frames per second
    frame_skip = int(video_fps / frames_per_second)

    # Create an empty array to store the pose keypoints for the video frames

    keypoints_data = []
    frame_index = 0
    for i in range(total_frames_video):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break
        
       
        orig_img = frame.copy()
        height_size = 256
        cpu = True
        track = 1
        smooth = 1
        net = net.eval()
        if not cpu:
            net = net.cuda()
        stride = 8
        upsample_ratio = 4
        num_keypoints = Pose.num_kpts
        previous_poses = []
        delay = 1

        img = orig_img
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses

        keypoints_data.append(current_poses)

        for pose in current_poses:
            pose.draw(img)

        if track:
            for pose_id, pose in enumerate(current_poses):
                cv2.putText(img, 'id: {}'.format(pose_id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))

        #cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        #time.sleep(0.2)
        #if cv2.waitKey(delay) == 27:
            #break

        frame_index += frame_skip

    cap.release()
    cv2.destroyAllWindows()


    max_persons = 5
    num_frames = min(len(keypoints_data), 24)  # Limit the number of frames to 24
    num_keypoints = Pose.num_kpts
    output_data = np.zeros((max_persons, num_keypoints, 2, 24))


    for i in range(num_frames):
        poses = keypoints_data[i]  # Capture every other frame
        for j in range(min(len(poses), max_persons)):
            keypoints = poses[j].keypoints
            output_data[j, :, :, i ] = keypoints[:, 0].reshape((num_keypoints, 1))  # Assign x-coordinates
            output_data[j, :, :, i ] = keypoints[:, 1].reshape((num_keypoints, 1))  # Assign y-coordinates

    return output_data





# Usage example:
"""video_path = '/home/nuno/Documents/GitHub/heimdall/Videos/corner-kick-blue/2023-06-26 15:14:22.626207output.mp4'  # Provide the actual video path
skeleton_data = process_video(video_path)
for skeleton in skeleton_data:
    print('skelet', skeleton.shape)"""


"""import os

video_folder = '/home/nuno/Downloads/FliptotalNew'  # Replace with the path to your video folder

data = []  # List to store skeleton data
labels = []  # List to store corresponding labels
count = 1
for class_folder in os.listdir(video_folder):
    class_path = os.path.join(video_folder, class_folder)
    if not os.path.isdir(class_path):
        continue

    class_name = class_folder  # Assuming the folder name represents the class name

    for video_file in os.listdir(class_path):
        count += 1
        print(count)
        video_path = os.path.join(class_path, video_file)
        if not video_file.endswith('.mp4'):
            continue

        # Process the video and obtain skeleton data (replace with your own code)
        skeleton_data = process_video(video_path)
        print(skeleton_data.shape)

        # Append the skeleton data and class label to the respective lists
        data.append(skeleton_data)
        labels.append(class_name)

# Now you have the skeleton data in the 'data' list and corresponding labels in the 'labels' list
# You can further process or save this information as needed
for i in range(len(data)):
    print('Class:', labels[i])
    for skeleton in data[i]:
        print('Skeleton:', skeleton)

# Save the data and labels
np.save('skeleton_data_filesflip.npy', data)
np.save('labels_posenet_filesflip.npy', labels)
"""


# for Joeys data
import os
video_folder_path = '/home/nuno/Downloads/referee_videos'
text_file_path = '/home/nuno/Downloads/animations_played.txt'
video_paths = []
video_labels = []
count = 0
# Read the text file
with open(text_file_path, 'r') as file:
    for line in file:
        count += 1
        if count == 1: #skip video 0, because it is not a video
            continue
        if count == 2000:
            break
        line = line.strip()
        if line:
            # Split the line into number and label
            number, label = line.split(':')

            # Remove the last part from the label
            label = label.rsplit('_Team_mcp', 1)[0].strip()

            # Construct the video path
            video_path = os.path.join(video_folder_path, f'out{number}.mp4')

            # Append the video path and label to the respective lists
            video_paths.append(video_path)
            video_labels.append(label)

data = []  # List to store skeleton data
labels = []  # List to store corresponding labels
count = 0
for video, label in zip(video_paths, video_labels):
    #check if the video exists
    if not os.path.exists(video):
        continue
    print(video, label)
    # Process the video and obtain skeleton data (replace with your own code)
    skeleton_data = process_video(video)
    count += 1
    print(count)
    print(skeleton_data.shape)
    # Append the skeleton data and class label to the respective lists
    data.append(skeleton_data)
    labels.append(label)

# Convert the 'data' list into a NumPy array
data = np.array(data)
labels = np.array(labels)

print(data.shape)
# Now you have the skeleton data in the 'data' list and corresponding labels in the 'labels' list
np.save('skeleton_data_posenet_ref_all.npy', data)
np.save('labels_posenet_ref_all.npy', labels)


