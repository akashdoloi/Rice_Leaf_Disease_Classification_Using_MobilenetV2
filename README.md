
# Rice Leaf Disease Classification Using MobileNetV2

A **Flask-based web application** for detecting rice leaf diseases using the **MobileNetV2** deep learning model.  
Simply upload a leaf image, and the app will classify it into one of the following categories:
- **Brown Spot**
- **Leaf Blast**
- **Bacterial Blight**
- **Sheath Blight**
- **Healthy**

### Key Features
- **High Accuracy:** Achieves **91.8% test accuracy** using transfer learning and fine-tuning.
- **Lightweight Model:** Optimized for **IoT devices** and **Raspberry Pi** deployment.
- **Smart Agriculture Ready:** Enables early detection of diseases to assist farmers in crop management.

### Tech Stack
- **Backend:** Flask (Python)
- **Model:** MobileNetV2 (TensorFlow/Keras)
- **Frontend:** HTML, CSS, Bootstrap
- **Deployment:** Local server or IoT edge devices

### Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Rice_Leaf_Disease_Classification_Using_MobilenetV2.git
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   python app.py
   ```

4. Open the app in your browser at:

   ```
   http://127.0.0.1:5000
   ```

### Future Improvements

* Integration with **real-time camera input** for continuous monitoring.
* Support for **EfficientNet-Lite** for even better IoT performance.
* Multi-language support for farmer-friendly interfaces.

---

**Author:** Akash Doloi
**Accuracy:** 91.81%
**Purpose:** Promoting smart agriculture through AI-powered disease detection.

