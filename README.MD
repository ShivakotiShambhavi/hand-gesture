# 🤖 Hand Gesture Recognition System

A real-time Hand Gesture Recognition system using **OpenCV** and **MediaPipe**, this project accurately detects and counts the number of fingers displayed in a webcam feed. It leverages background modeling, contour analysis, and convex hull geometry for gesture analysis.
This project is a great foundation for gesture-based interfaces, sign language interpretation, and touchless control applications.

---

## 📷 Demo picture
![image alt](https://github.com/RAVULAAJAY/Hand-Gesture-Recognition/blob/6345855aafe966f48cb6e9d45f405fc3554cd79d/images/IMG-20250529-WA0007.jpg)

---

## 🔑 Key Features

* ✅ Real-time gesture recognition
* ✅ Background calibration and segmentation
* ✅ Finger counting via convex hull and contour heuristics
* ✅ ROI-based hand detection
* ✅ Optional MediaPipe integration for enhanced hand tracking

---

## 🧰 Tech Stack

* **Language**: Python 3.10+
* **Core Libraries**:

  * OpenCV
  * NumPy
  * scikit-learn (for distance calculation)
  * imutils
  * MediaPipe *(optional for advanced tracking)*

---

## ⚙️ Getting Started

### 🔽 1. Clone the Repository

```bash
git clone https://github.com/yourusername/hand-gesture-recognition.git
cd hand-gesture-recognition
```

### 📦 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 🚀 3. Run the Application

```bash
python recognize.py
```

> Make sure your webcam is connected and accessible.

---

## 🧠 How It Works

1. **Background Initialization** – Averages the first 30 frames to estimate the static background.
2. **Segmentation** – Subtracts the background from current frame and thresholds it to isolate the hand.
3. **Convex Hull Analysis** – Computes extreme points of the hand to form a circular ROI.
4. **Finger Counting** – Analyzes contours within the circular ROI to count extended fingers.

---

## 🗂️ Directory Structure

```plaintext
hand-gesture-recognition/
├── recognize.py             # Main script for recognition
├── requirements.txt         # Python dependencies
├── README.md                # Documentation (this file)
├── images/                  # Output screenshots
│   ├── demo1.png
│   ├── demo2.png
└── .gitignore               # Ignore cache and logs
```

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 🔮 Future Scope

* ⬜ Replace OpenCV-only pipeline with MediaPipe holistic tracking
* ⬜ Incorporate ML/DL classifiers for custom gestures
* ⬜ Build a user-friendly GUI using Tkinter or PyQt
* ⬜ Multi-user / multi-hand support

---

## 👤 Author

**Shivakoti Shambhavi**
🔗 [GitHub Profile](https://github.com/dashboard)
---
> ✨ This project is currently under active development.
