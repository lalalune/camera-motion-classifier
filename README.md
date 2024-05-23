# Camera Motion Classifier

This is a small example script to demonstrate how to use the `CameraMotionClassifier` class to classify camera motion in a video file. The script reads a video file and classifies the camera motion in each frame. The camera motion is classified into one of the following categories:
STATIC, ORBIT_LEFT, ORBIT_RIGHT, ORBIT_UP, ORBIT_DOWN, PAN_LEFT, PAN_RIGHT, CRANE_UP, CRANE_DOWN, TILT_LEFT, TILT_RIGHT, TILT_UP, TILT_DOWN, ZOOM_IN, ZOOM_OUT, DOLLY_IN, DOLLY_OUT

To run:
```
python camera_motion_classifier.py
```

To predict a video:
```
python camera_motion_classifier.py --predict example.mp4
```