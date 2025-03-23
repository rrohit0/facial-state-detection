# 🚀 Drowsiness Detection System

## 📌 Project Overview
This project is a **real-time drowsiness detection system** that uses **Mediapipe for face detection** and a **deep learning model** to predict whether a driver is drowsy. If the system detects that the driver's eyes are closed for more than **2 seconds**, it **triggers an alert sound** to wake them up.

## ⚙️ Features
✅ **Real-time face detection** using **Mediapipe**  
✅ **Deep Learning-based eye state prediction**  
✅ **Alert system that plays a warning tone if drowsiness is detected**  
✅ **Fast and efficient performance** optimized for real-time use  
✅ **Customizable alert settings** (MP3 sound file, detection threshold, etc.)  

---

## 📥 Installation & Setup

### **1️⃣ Create & Activate a Virtual Environment**
#### **On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

#### **On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### **2️⃣ Install Required Dependencies**
```bash
pip install -r requirements.txt
```
If `requirements.txt` does not exist, install manually:
```bash
pip install opencv-python mediapipe tensorflow numpy pygame argparse json
```

### **3️⃣ Prepare Model Files**
- Place your **trained model** (`facial_state_model.h5`) in the project directory.
- **No class mapping file is required**, as the model directly predicts states.

### **4️⃣ Run the System**
To start the drowsiness detection system, run:
```bash
python enhanced-monitoring-system.py
```

To run with a custom alert sound:
```bash
python enhanced-monitoring-system.py --alert-sound "path/to/your_alert.mp3"
```

---

## 🎯 How It Works
1. **Face detection**: Mediapipe detects the face in the video stream.
2. **Eye & mouth state prediction**: The deep learning model classifies whether the **eyes are open or closed** and if the **driver is yawning**.
3. **Drowsiness detection**: If the driver's **eyes remain closed for more than 2 seconds**, an **alert tone** is played.

---

## ⚙️ Customization
### **Change the Alert Sound**
- Replace the **default alert tone** (`alert_tone.mp3`) with your own sound file.
- Run the script with a custom sound file:
  ```bash
  python enhanced-monitoring-system.py --alert-sound "path/to/your_alert.mp3"
  ```

### **Modify the Drowsiness Detection Threshold**
- Default **eye closure duration** is **2 seconds**.
- To adjust it, modify the `monitor_alertness` function in `enhanced-monitoring-system.py`.

---

## 🛠️ Possible Enhancements
🚀 **Improve accuracy** by training the model on a larger dataset.  
🎯 **Use a better face detection model** like MediaPipe's Face Mesh.  
📲 **Deploy as a mobile app** using TensorFlow Lite.

---

## 🤝 Contributing
Feel free to **fork the repo**, create a **pull request**, or open **issues** for suggestions!

---

## 📧 Contact
For any queries, reach out at **rcrathod13@gmail.com** or open an issue on GitHub!

Happy Coding! 🚀

