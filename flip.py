import os
import cv2


# function to flip all videos in a folder
def flip_videos_in_folder(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over the files in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Check if the file is a video
        if os.path.isfile(input_path) and filename.endswith(('.mp4', '.avi', '.mkv')):
            # Open the video file
            video = cv2.VideoCapture(input_path)
            
            # Get video properties
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(video.get(cv2.CAP_PROP_FPS))
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create a new video writer
            output_path = os.path.join(output_folder, filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Read and flip each frame of the video
            for frame_num in range(total_frames):
                ret, frame = video.read()
                
                if not ret:
                    break
                
                # Flip the frame horizontally
                # flipped_frame = cv2.flip(frame, 1)

                if "PXL" in filename or "VID" in filename:
                    flipped_frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif "IMG" in filename:
                    flipped_frame = frame
                # Write the flipped frame to the output video
                output_video.write(flipped_frame)
            
            video.release()
            output_video.release()
            print(f"Flipped video saved: {output_path}")




# loop through all the videos in folders in a folder

# run on all folders for Nuno
# for folder in os.listdir('/home/nuno/Documents/universiteit/tweedejaarsproject/real/FlipTotal'):
#     print(folder)
#     flip_videos_in_folder(f'/home/nuno/Documents/universiteit/tweedejaarsproject/real/FlipTotal/{folder}', f'/home/nuno/Documents/universiteit/tweedejaarsproject/real/NewFliptotal/{folder}')

# Run on all folders for Fiona
for folder in os.listdir('/home/fiona/Documents/TweedeJaarsProject/FlipTotal'):
    print(folder)
    flip_videos_in_folder(f'/home/fiona/Documents/TweedeJaarsProject/FlipTotal/{folder}', f'/home/fiona/Documents/TweedeJaarsProject/NewFliptotal_2/{folder}')


