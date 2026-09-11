# Convert video frames into a stereo XY oscilloscope signal.

Works best with an analog scope:

Left audio channel  -> X (oscilloscope Channel I, "black")
Right audio channel -> Y (oscilloscope Channel II, "red")

Image processing pipeline:

    grayscale
    optional CLAHE
    Gaussian blur
    Sobel X/Y gradients
    Sobel magnitude threshold
    Canny edge detector
    morphology closing
    contour extraction
    polygon simplification
    temporal contour matching and stable scan ordering
    arc-length resampling

The resulting XY trajectory is repeated at --trace-hz and encoded as
44.1 kHz stereo audio. It can be written to a WAV file and/or played
through the sound card in real time.

Dependencies:

    pip install numpy opencv-python

Optional for --play:

    pip install sounddevice

Usage:

    For a video file, a good first experiment is:

    python video_to_xy_audio.py input.mp4 -o scope.wav --preview

    Then play scope.wav normally with all sound enhancements/EQ disabled. Left goes to HM507 CH I/X, right to CH II/Y.

    For direct real-time playback:

    python video_to_xy_audio.py input.mp4 --play --preview

    or from a webcam:

    python video_to_xy_audio.py 0 --play --preview --fps 30

    The parameters I would experiment with first are:

    python video_to_xy_audio.py input.mp4 -o scope.wav --preview \
        --trace-hz 60 \
        --sobel-percentile 72 \
        --blur 5 \
        --max-contours 32 \
        --epsilon 0.004 \
        --min-perimeter 18    
