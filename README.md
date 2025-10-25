Vehicle Attribute Identification System

This project is an **AI-powered Vehicle Attribute Detection System** built using **YOLOv8**.  
It identifies vehicles, license plates, and brand logos from images or live video feeds — and provides per-vehicle insights like:
- Lane position (Left / Center / Right)
- Plate presence (`Yes` / `No`)
- Vehicle direction (`Incoming` / `Outgoing`)
- Real-time statistics on total vehicles, with/without plates, etc.

The system combines **Ultralytics YOLOv8** for detection with a **FastAPI backend** for inference and a **Streamlit** dashboard for visualization.

---

## 📁 Project Structure

vehicle_attribute_identification/
│
├── api/
│ ├── model.py # YOLOv8 model loading and inference logic
│ ├── main.py # FastAPI app exposing /predict endpoint
│ ├── outputs/ # Annotated result images are saved here
│
├── streamlit_app.py # Streamlit frontend UI
├── requirements.txt # All dependencies
├── data.yaml # YOLO dataset configuration file
├── yolov8_vehicle_model/ # Trained YOLOv8 model weights
│ └── best.pt
├── README.md # This file
└── ...


---

## 🧠 Model Overview

- **Base Model:** YOLOv8 (fine-tuned)
- **Trained on:** Custom dataset of vehicles and license plates
- **Classes:** 14 total

```yaml
nc: 14
names:
  [ 'truck', 'license_plate', 'logo', 'car', 'bike', 'bus',
    'nissan', 'toyota', 'isuzu', 'mitsubishi', 'proton',
    'mazda', 'mercedes', 'honda' ]
⚙️ Installation
Clone the repository


git clone https://github.com/<your_username>/vehicle-attribute-identification.git
cd vehicle-attribute-identification
Create a virtual environment


python -m venv venv
source venv/bin/activate     # on Linux/Mac
venv\Scripts\activate        # on Windows
Install dependencies

pip install -r requirements.txt
Set up model path
Make sure MODEL_PATH in api/model.py points to your trained YOLOv8 weights:

python
Copy code
MODEL_PATH = "path/to/yolov8_vehicle_model/weights/best.pt"
🚀 Running the System
1️⃣ Start the FastAPI Backend
bash
Copy code
uvicorn api.main:app --reload
The API will be live at: http://127.0.0.1:8000

Test it via Swagger UI: http://127.0.0.1:8000/docs

/predict Endpoint
Request:

bash
Copy code
POST /predict
Content-Type: multipart/form-data
Body:

file: image file (jpg/png)

Response Example:

json
Copy code
{
  "summary": {
    "timestamp": "2025-10-24 22:09:15",
    "vehicle_count": 4,
    "vehicles_with_plate": 2,
    "vehicles_without_plate": 2,
    "incoming": 1,
    "outgoing": 3,
    "detections": [
      {
        "class": "car",
        "conf": 0.98,
        "bbox": [234, 115, 580, 400],
        "plate_status": "Yes",
        "lane": "Center",
        "direction": "Incoming"
      }
    ]
  },
  "annotated_image_base64": "<base64_string>"
}
2️⃣ Run the Streamlit Dashboard
bash
Copy code
streamlit run streamlit_app.py
Upload any image to visualize detections.

The dashboard displays:

Annotated image with bounding boxes.

Real-time stats (vehicles, plates, direction, lanes).

Option to download annotated results.

🧩 Key Features
✅ Color-preserving inference — no grayscale conversion
✅ Per-vehicle plate detection — clearly marked “Plate: Yes/No”
✅ Lane estimation — Left / Center / Right
✅ Direction tagging — Incoming / Outgoing
✅ Top-bar summary — instant vehicle statistics
✅ Robust plate–vehicle matching using IoU + centroid logic
✅ Configurable thresholds (CONF_THR, IOU_ASSOC_THRESH) in model.py
✅ Output to both file and base64 for web integration

📊 Example Output
Feature	Description
🟩 Green Boxes	Vehicles (Incoming)
🟥 Red Boxes	Vehicles (Outgoing)
🟨 Yellow Boxes	License Plates
🟧 Orange Label	Plate Status
🕹️ Header	Stats: Vehicle count, Plates, Direction summary

🧪 Improving Model Accuracy
If the model performs poorly on new web images:

Use Transfer Learning: fine-tune from yolov8m.pt or yolov8l.pt.

Increase Resolution: train and infer with imgsz=1024.

Add Augmentations: flips, color jitter, blur, mosaic.

Evaluate with TTA: use augment=True during inference.

Recommended command:

yolo detect train model=yolov8m.pt data=path/to/data.yaml epochs=200 imgsz=1024 batch=8 augment=True
🧰 Tech Stack
Component	Tool / Library
Object Detection	Ultralytics YOLOv8
Backend API	FastAPI
Frontend UI	Streamlit
Image Handling	OpenCV, NumPy
Language	Python 3.10+

🧑‍💻 Developer Notes
All inference results (annotated images) are saved under api/outputs/.

Default confidence threshold is 0.25; tweak for sensitivity.

Direction estimation is heuristic — for video tracking, integrate model.track(persist=True).

🧾 License
This project is released under the MIT License.
You’re free to use, modify, and distribute — with attribution.

📬 Contact
For issues, improvements, or collaboration, open an issue or contact the developer:

Author: [Your Name]
Email: your_email@example.com
GitHub: https://github.com/<your_username>

yaml
