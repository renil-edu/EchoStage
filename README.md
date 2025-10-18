# EchoStage
HackTX 2025

***

## Example README Installation Section

### Installation Instructions

This project requires Python 3.10+, Conda, and pip. Follow the steps below depending on your operating system.

***

### macOS / Linux

1. Open a terminal.

2. Create and activate the Conda environment:

```bash
conda create -n face_env python=3.10 -y
conda activate face_env
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install opencv-python deepface tensorflow-macos keras tf-keras protobuf==4.25.8
```

***

### Windows

1. Open Anaconda Prompt or Powershell.

2. Create and activate the Conda environment:

```powershell
conda create -n face_env python=3.10 -y
conda activate face_env
```

3. Install dependencies:

```powershell
pip install --upgrade pip
pip install opencv-python deepface tensorflow keras tf-keras protobuf==4.25.8
```

***

### Running the App

Once dependencies are installed, run your emotion detection app with:

```bash
python emotion_detector.py
```

***

### Notes

- `tensorflow-macos` is only for macOS; Windows uses `tensorflow` instead.
- Pinning protobuf to version 4.25.8 avoids common compatibility issues.
- Make sure your webcam is connected and accessible.
- Press ‘a’ to start scene recording, ‘s’ to stop and save, and ‘q’ to quit.
