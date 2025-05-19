# Virtual Glasses Try-On

This Python project uses OpenCV and MediaPipe to let users virtually try on different glasses in real-time using their webcam.

## Features

* Detects face landmarks using MediaPipe.
* Overlays transparent PNG glasses images onto the user's face.
* Switch between different glasses styles using the `n` key.
* Press `q` to quit the application.

## Requirements

Install the required packages:


pip install opencv-python mediapipe numpy


## Setup

1. Create a folder named `glasses` in the same directory as the script.
2. Add your transparent `.png` glasses images to the `glasses` folder.
3. Make sure the images are ordered by filename (e.g., `1.png`, `2.png`, ...).

## Usage

Run the script:

python your_script_name.py


* Press `n` to switch to the next pair of glasses.
* Press `q` to quit.

## Notes

* Make sure your webcam is working.
* PNG images must have an alpha (transparency) channel.
