# RefChall2023
Welcome to our Referee Challenge 2023 GitHub repository!

This repository contains our work on developing a CNN-LSTM model to classify movements. We began by using Movenet and PoseNet to extract skeleton keypoints over several seconds, which served as input for the LSTM model to classify different movements.

To create the necessary datasets, we filmed numerous videos using both phone cameras and NAO robots, then augmented these videos and generated synthetic data using Unity. The results of our analysis can be found in the accompanying PDF.

Our most promising outcome was an accuracy rate of nearly 90% on the phone camera data. Unfortunately, this performance did not transfer well to the NAO videos, but we believe that this can be improved with additional data.

This repository also includes data and model files related to phone camera data, flipped phone camera data, and filtered synthetic videos generated with Unity. Additionally, it contains model files for Movenet and data generated using OpenPose.

## Video Data

The data is organized into the following folders:

- [Phone Camera Data](https://drive.google.com/drive/folders/1Djf08R4_V_pmfwGo8ArrQP0Fh2_QT-L-?usp=sharing): This folder contains phone camera data captured for the our project.
-The video's are flipped to create more data using flip.py
- [Filtered Synthetic Data Videos](https://drive.google.com/drive/folders/1UrIsa4aTwmriCEGGBAUuCYwvyMIwRHoU?usp=sharing): This folder contains filtered synthetic data videos that were created using Unity.
- [Nao video's](https://drive.google.com/drive/folders/1QDIIl79lVmP6LSYIN92a7_ZuIUEZBuBS?usp=sharing): This folder contains the dataset we filmed on the NAO V6 robots

## Model Files

The model files for Movenet and OpenPose can be found in the following folder:

- [Model Files](https://drive.google.com/drive/folders/17rzZSiDrDhjIR0rv9Bt9Om3X43PPH-y-?usp=sharing): This folder contains the model files required to use Movenet. Additionally, you will find data created using OpenPose in the subfolder "OpenPose". Also other .npy files for our skeleton data can be found here. The folder 'padding' is the best working version using Movenet Thunder.

To use OpenposeCNN, you can clone the [Lightweight Openpose](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) repository and follow the instructions provided. Place the `OpenposeCNN.py` script in the main folder of the cloned repository.


Contributors:
Dario Xavier Catarrinho, Lasse van Iterson, Joey van der Kaaij, Fiona Nagelhout, Nils Peters, Nuno Scholten, Arnoud Visser

Feel free to contribute to this project by submitting pull requests or creating issues. If you have any questions or suggestions, please open an issue and we'll be happy to assist you.

Happy coding!



