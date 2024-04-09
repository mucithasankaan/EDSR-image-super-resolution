
# IMAGE-SUPER-RESOLUTION-SYSTEM

#### This project is about the fundamentals of super-resolution artificial intelligence, specifically centered on the Enhanced Deep Super-Resolution (EDSR) technique. EDSR is used to improve image quality and resolution. The project includes a practical example demonstrating how a basic EDSR model works in enhancing images.

## Prerequisites:
1: Python 3.8 or later <br>
2: Compatible with Windows, MacOS, Linux <br>
3: midrange to powerful computer

## Installation Instructions:

### If you have an RTX graphics card, you can use the CUDA system to significantly speed up artificial intelligence training. Instructions for setting up the CUDA system:

1: Please go to the website https://pytorch.org/, scroll down a bit, and retrieve the download command for a PyTorch version compatible with CUDA as per your requirements. Then, execute it in your Python environment's console.
![pytorch](ReadMe_images/pytorch.png)

2: Please go to the link https://developer.nvidia.com/cuda-toolkit-archive, find and download the version of CUDA you chose while installing the PyTorch library. For example, as shown in the photo above, I opted for version 11.8 and downloaded it from this archive.
![cuda archive](ReadMe_images/cuda_archive.png)

3: If you've completed all the steps correctly, run the Trainmodel code. If everything is as it should be, after running the code, it should say "Device: cuda". If it says "Device: cpu", something went wrong. In this case, uninstall the torch and torchvision libraries using pip: pip uninstall torch torchvision; if you're using conda: conda remove pytorch, conda remove torchvision. Then, reinstall them using the code from this link: https://pytorch.org/.

### If you do not have an RTX graphics card, you can install the libraries in the usual way:
1 pip: <br>
pip install torch torchvision

2 conda: <br>
conda install pytorch torchvision -c pytorch

## Included Codes:
### Video2Frame_Data_Generator.py:
This code randomly selects a specified total number of frames from videos in the video_datas folder, then creates two different images with those frames at two different qualities: low quality at 720p and high quality at 1080p. It saves these images in the 720p_frames and 1080p_frames folders for the purpose of training artificial intelligence.

### TrainModel.py:
If you've completed all the steps correctly, run the Trainmodel code. If everything is as it should be, after running the code, it should say "Device: cuda". If it says "Device: cpu", something went wrong. In this case, uninstall the torch and torchvision libraries using pip: pip uninstall torch torchvision; if you're using conda: conda remove pytorch, conda remove torchvision. Then, reinstall them using the code from this link: https://pytorch.org/.

### inference_code:
This code loads a trained and saved model, and attempts to enhance the quality and resolution of a given photograph using this model, then saves the enhanced photograph.


## Contact
mucithasankaan@gmail.com <br>
https://instagram.com/mucithasankaan
