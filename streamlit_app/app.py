import streamlit as st
import requests
from PIL import Image
import io
import base64

# Page configuration
st.set_page_config(
    page_title="SolvAI Vehicle Detector",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme and styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1a202c;
        color: #e2e8f0;
    }
    .stButton>button {
        background-color: #48bb78;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #38a169;
    }
    .stFileUploader {
        background-color: #2d3748;
        padding: 20px;
        border-radius: 5px;
        text-align: center;
    }
    .stImage {
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stSuccess {
        color: #48bb78;
    }
    .stError {
        color: #f56565;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title("🚗 SolvAI Vehicle Detector")
st.markdown("Upload a vehicle image, and the app will detect its attributes (class, plates, logos) using your trained YOLOv8 model.")

# File uploader
uploaded_file = st.file_uploader("Upload an image (JPG/PNG, max 20MB)", type=["jpg", "jpeg", "png"], help="Drag and drop or click to select an image.")

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Run Detection"):
        with st.spinner("Detecting attributes with SolvAI AI..."):
            # Send to FastAPI backend
            files = {"file": uploaded_file.getvalue()}
            try:
                response = requests.post("http://127.0.0.1:8000/predict", files=files, timeout=30)

                if response.status_code == 200:
                    data = response.json()

                    # Display detection summary
                    st.success(f"✅ Detected {data['num_detections']} vehicle(s)")
                    st.markdown(f"**Vehicles with Plates:** {data['summary']['vehicles_with_plate']} | **Vehicles with Logos:** {data['summary']['vehicles_with_logo']}")

                    # Display annotated image from backend
                    if data.get("annotated_image_base64"):
                        annotated_image = Image.open(io.BytesIO(base64.b64decode(data["annotated_image_base64"])))
                        st.image(annotated_image, caption="Detection Results", use_container_width=True)

                    # Display detailed detections in a table
                    if data["detections"]:
                        st.subheader("Detection Details")
                        detection_data = []
                        for det in data["detections"]:
                            detection_data.append([
                                det["class"],
                                f"{det['conf']:.2f}",
                                det["plate_status"],
                                det["logo_status"],
                                det["lane"],
                                det["direction"]
                            ])
                        st.table({
                            "Class": [row[0] for row in detection_data],
                            "Confidence": [row[1] for row in detection_data],
                            "Plate": [row[2] for row in detection_data],
                            "Logo": [row[3] for row in detection_data],
                            "Lane": [row[4] for row in detection_data],
                            "Direction": [row[5] for row in detection_data]
                        })
                    else:
                        st.warning("No specific detections found.")
                else:
                    st.error(f"Error: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {str(e)}. Ensure the backend is running at http://127.0.0.1:8000/predict.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built by SolvAI | © 2025", help="A team project for vehicle attribute detection.")