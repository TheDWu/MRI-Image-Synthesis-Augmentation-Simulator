# Medical Image Synthesis and Augmentation Simulator

## Introduction
In this project, we present a medical image synthesis and augmentation simulator. The simulator generates undersampled images from high-quality images using algorithms and a Graphical User Interface (GUI) developed in Python. The implemented algorithms include image generation and two types of modifications on the images, with the simulator's results showcased.

## Objectives
1. **MRI Acquisition:** Simulate MRI data collection.
2. **K-Space Truncation:** Perform K-Space truncation on the original image.
3. **Gaussian Noise Addition:** Apply Gaussian noise addition to the K-Space truncated image.
4. **Image Augmentation:** Generate augmented images from the original image.
5. **Final Processed Image Generation:** Create final images with both K-Space truncation and Gaussian noise.
6. **Graphical User Interface:** Design and implement a GUI capable of executing the previous objectives.


## How it works: 
The image processor application is a Python script that provides a graphical user interface for loading, processing, and visualizing images. It utilizes the Tkinter library for the GUI, OpenCV for image processing, and other modules for additional functionalities. 

## Prerequisites:

Before running the application, the user has to make sure the required Python modules installed. You can install them using the following commands:

Python version 3.10.6

Libraries: Numpy , opencv-python, matplotlib, Tkinter SciPy, Pillow 
pip install numpy v1.26
pip install matplotlib v3.8
pip install pillow v10.1.0
pip install opencv-python  v4.8.1.78
pip install scipy v1.11.4
pip install albumentations v23.2.1
pip install tkinter v23.3.1


Running the application:

To run the Image Processor application, execute the following command in your terminal or command prompt:

“python main.py”

This will launch the application, and you can follow the on-screen instructions to load, process, and visualize images. 


## Functionality:

Loading an Image:
Click the "Load Image" button to open a file dialog and select an image (PNG, JPG, JPEG). The loaded image will be displayed in the main window.

Processing an Image:
Click the "Generate Gaussian Image" button to apply Gaussian noise to the loaded image. The processed image will be displayed in a new window.

Augmenting Images
Click the "Generate Augmented Images" button to generate a set of augmented images using various transformations. The augmented images will be displayed in a new window.

K-Space Truncation
Click the "Generate K-Space Truncation Images" button to perform k-space truncation on the loaded image. The resulting images will be displayed in a new window.

Combined Process
Click the "Generate Final Image" button to combine Gaussian noise addition and k-space truncation for different downsampling factors. The combined processed images will be displayed in a new window.

## Walkthrough:

1. To navigate to the project file in your favorite IDE, open the IDE and look for a "File" menu at the top. Click on "File" and then select "Open" or "Open Project." Browse to the location where your project file is stored, select the file, and click "Open" to load the project into the IDE

2. The user can then enter "python main.py" in the terminal to launch the program. 

3. For the user to load the photo into the interface the user must press the “Load Image” button once. After that, the user can find the picture they want to use in the graph.

4.The user can now select the process they want to carry out once the image has been imported into the graph. The Generate Final Image, Generate Gaussian Image, Generate Augmented Images, and Generate K-space Truncation Image buttons are available for selection by the user.
## Files included

**Ground Truth Folder**: Contains sample brain MRI's that are loaded into the simulator

**main.py**: The file with the code 

**Presentation Video**: Demo the simulator in action

**MRI Image Simulator Project Report**: A detailed report of the project objectives, methods, and algorithms



