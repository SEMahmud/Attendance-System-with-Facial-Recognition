# Facial Recognition Attendance System

A small Python attendance application that identifies known people from a webcam feed and records recognized names in a CSV file.

## Architecture

The application runs as a single local Python process:

1. **Reference images** in `ImagesAttendance/` are loaded at startup. Each image filename (without its extension) is used as the person's name.
2. **Face encoding** converts each reference image into a facial encoding using `face_recognition`.
3. **Live recognition** reads frames from the default webcam through OpenCV, scales each frame down, detects faces, and compares their encodings against the known encodings.
4. **Display and attendance** draws a label around detected faces in the webcam window and appends a row to `Attendance.csv` when a recognized name has not already been recorded.

`AttendanceProject.py` is the runnable application. `Attendance.ipynb` and `Basics.py` contain exploratory image loading, face encoding, and face-comparison examples; they are not required to run attendance capture.

## Libraries

- [OpenCV (`opencv-python`)](https://pypi.org/project/opencv-python/) for webcam capture, image processing, and display.
- [NumPy](https://pypi.org/project/numpy/) for numerical operations on face distances.
- [face_recognition](https://pypi.org/project/face-recognition/) for face detection, facial encodings, and matching. It depends on `dlib`.
- Python standard library: `os` for image paths and `datetime` for attendance timestamps.

## Quickstart

Use Python 3.10 or 3.11 and run these commands in PowerShell from the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install opencv-python numpy face_recognition
python AttendanceProject.py
```

On Windows, installing `face_recognition` may require a compatible `dlib` wheel or native build tools. If installation fails, install a `dlib` build compatible with your Python version first, then retry `face_recognition`.

## Data Setup

- Put one clear image per person in `ImagesAttendance/`. The filename should match the displayed name, for example `Raihan Sikdar.jpg`.
- The script expects to find a face in every reference image and uses the first detected face. Invalid images or images with no detectable face can stop startup.
- Connect a webcam and allow the application to access it. The application uses camera index `0`.
- Run the script from the project directory so the relative image and CSV paths resolve.
- Attendance is appended to `Attendance.csv`. The script includes ID and section mappings for names in `AttendanceProject.py`; review those mappings before using it with a different roster.

The camera window runs until the Python process is interrupted (for example, with `Ctrl+C` in the terminal).
