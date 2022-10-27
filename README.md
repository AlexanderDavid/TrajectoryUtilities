# Trajectory Utilities

A collection of common utilities operating on files of (normally) human trajectories. These trajectories are expected in a CSV format with columns for the id, timestamp, x, y, goal x, and goal y positions in that order. 

## Video Rendering
```
usage: render_video.py [-h] [--fps FPS] [--save SAVE] input_file

Render a trajectory file to an mp4

positional arguments:
  input_file   Path to the trajectory file

optional arguments:
  -h, --help   show this help message and exit
  --fps FPS    Frames per second to render at
  --save SAVE  Path to save the animation to
```