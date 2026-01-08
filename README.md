# TinyML Apnea Detection on ESP32-S3 

**Project:** Real-time Obstructive Sleep Apnea (OSA) detection using audio signals on the ESP32-S3.
**Method:** Custom **Inception Lite** deep learning model trained with TensorFlow and deployed via Edge Impulse.

## Repository Contents

| File | Description |
| :--- | :--- |
| **`apnea_detection_microphone.ino`** | Main firmware for ESP32-S3. Handles I2S audio sampling, inference, and alert logic. |
| **`train_inception_lite_esp32.py`** | Python script for training the custom Inception Lite model (Keras/TF). |
| **`ei-apnea-detection-esp32.zip`** | Edge Impulse C++ library export (contains the quantized model and DSP). |
| **`inception_lite_architecture.png`** | Diagram of the Neural Network architecture. |
| **`block_diagram.png`** | Hardware connection diagram. |
| **`Flowchart.png`** | Firmware logic flowchart. |


## Hardware Requirements
* **MCU:** ESP32-S3 (DevKitC-1 or similar).
* **Sensor:** INMP441 I2S Microphone.
* **Pinout:** SCK (14), WS (15), SD (16).

## Resources

* **Demo Video:** [Click here to watch the demo](https://drive.google.com/drive/u/0/folders/1jF2YqXcGuUXTcJqnaTetIAsx0LQMYZwj)
* **Dataset:** [Link to Dataset](https://drive.google.com/drive/folders/13t2_g_KirX5G2_hlaG7Sr-WJ2Wh4ard-?usp=drive_link)

## Quick Start
1.  Download and extract `ei-apnea-detection-esp32.zip` to your Arduino `libraries` folder.
2.  Open `apnea_detection_microphone.ino` in Arduino IDE.
3.  Select **ESP32S3 Dev Module** (Enable OPI PSRAM).
4.  Upload and open the Serial Monitor.

---
*Author: Dang-Minh-Duong*
