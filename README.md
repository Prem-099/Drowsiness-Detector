# 💤 Drowsiness Detection System

A computer vision–based project that detects driver drowsiness in real-time using a webcam.
It alerts the user when signs of sleepiness (like eye closure or yawning) are detected.

## 📌 Features

- Real-time face and eye detection using OpenCV
- Detects drowsiness by monitoring eye aspect ratio (EAR)
- Plays an alarm sound when drowsiness is detected
- Lightweight and runs on most laptops with a webcam
- Can be extended for vehicle safety systems

## 🛠️ Tech Stack

- Python 3.9+

- OpenCV (for image processing)

- Dlib (for facial landmarks)

- NumPy (for calculations)

- Pygame (for alarm sounds)
---

## 📂 Project Structure
```bash
Drowsiness-Detection/
│── dataset/                # (optional) sample face/eye dataset
│── models/                 # pre-trained models (dlib, mediapipe, etc.)
│── alarm.wav               # alert sound
│── drowsiness_detect.py    # main script
│── requirements.txt        # dependencies
│── README.md               # project documentation
```

## 🚀 Installation & Usage
### 1. Clone the repository
```bash
git clone https://github.com/your-username/Drowsiness-Detection.git
cd Drowsiness-Detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the project
```bash
python drowsiness_detect.py
```

## ⚙️ How It Works

- Captures frames from the webcam.

- Detects face and eye landmarks.

- Calculates the Eye Aspect Ratio (EAR):

- If EAR < threshold for several frames → Eyes are closing.

- Triggers an alarm sound when drowsiness is detected.

## 🔮 Future Improvements

- Yawning detection using mouth aspect ratio

- Integration with car dashboard systems

- Mobile app version with TensorFlow Lite

## 🤝 Contributing

- Pull requests are welcome! For major changes, please open an issue first to discuss.

## 📜 License

- This project is licensed under the MIT License.
