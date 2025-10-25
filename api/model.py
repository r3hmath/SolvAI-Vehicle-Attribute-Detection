# api/model.py
import io
import os
import cv2
import base64
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# CONFIG
MODEL_PATH = r"C:\Users\fayaf\OneDrive\Desktop\Q2_Rahmath\Vehicle_Training\yolov8_vehicle_model\weights\best.pt"
CONF_THR = 0.25
IOU_ASSOC_THRESH = 0.25
IMGSZ = 1024  # inference size (increase for plates)
OUTPUT_DIR = os.path.join("api", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model (keep this in file)
_model = None
def load_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model

# utilities
def iou(boxA, boxB):
    # box = [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interA = interW * interH
    aA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    bA = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    denom = float(aA + bA - interA + 1e-9)
    return interA / denom if denom > 0 else 0.0

def nms_boxes(boxes, scores, classes, iou_thresh=0.45):
    # boxes Nx4, scores N, classes N -> do per-class NMS
    keep = []
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    for cls in np.unique(classes):
        idxs = np.where(classes == cls)[0]
        if len(idxs) == 0: continue
        b = boxes[idxs]
        s = scores[idxs]
        order = s.argsort()[::-1]
        while order.size > 0:
            i = order[0]
            keep.append(int(idxs[i]))
            ious = np.array([iou(b[i], bb) for bb in b])
            rem = np.where(ious <= iou_thresh)[0]
            order = order[rem]
    return keep

def inference(model, image_bytes, use_tta=True, return_b64=True, require_plate_class_name="license_plate"):
    """
    model: YOLO object
    image_bytes: bytes
    use_tta: whether to use augment=True in predict (recommended)
    returns: dict with 'summary' and either annotated_image_path or base64
    """
    # read image bytes to cv2 BGR
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")

    h, w = img.shape[:2]

    # Run model predict with TTA (augment=True) for better generalization
    predict_kwargs = {"source": img, "conf": CONF_THR, "imgsz": IMGSZ, "verbose": False}
    if use_tta:
        predict_kwargs["augment"] = True

    results = model.predict(**predict_kwargs)
    r = results[0]

    # Collect raw boxes
    boxes = []
    scores = []
    classes = []
    names = model.names

    # r.boxes.xyxy may be tensor-like; handle gracefully
    if hasattr(r, "boxes") and len(r.boxes) > 0:
        for b in r.boxes:
            xy = b.xyxy[0].cpu().numpy() if hasattr(b.xyxy[0], "cpu") else np.array(b.xyxy[0])
            conf = float(b.conf[0]) if hasattr(b, "conf") else float(b.conf)
            cls_idx = int(b.cls[0]) if hasattr(b, "cls") else int(b.cls)
            boxes.append([float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])])
            scores.append(conf)
            classes.append(cls_idx)

    # Nothing detected
    if len(boxes) == 0:
        annotated_path = os.path.join(OUTPUT_DIR, f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(annotated_path, img)
        summary = {
            "timestamp": datetime.now().isoformat(),
            "num_detections": 0,
            "vehicle_count": 0,
            "vehicles_with_plate": 0,
            "vehicles_without_plate": 0,
            "vehicles_with_logo": 0,
            "vehicles_without_logo": 0,
            "incoming": 0,
            "outgoing": 0,
            "detections": []
        }
        out = {"summary": summary, "annotated_image_path": annotated_path}
        if return_b64:
            with open(annotated_path, "rb") as f:
                out["annotated_image_base64"] = base64.b64encode(f.read()).decode("utf-8")
        return out

    # Run per-class NMS to remove duplicates
    keep_idx = nms_boxes(boxes, scores, classes, iou_thresh=0.45)
    boxes = [boxes[i] for i in keep_idx]
    scores = [scores[i] for i in keep_idx]
    classes = [classes[i] for i in keep_idx]

    # Separate vehicle classes vs plates/logos by name
    plate_names = {require_plate_class_name}
    vehicle_name_set = set(["car", "truck", "bus", "bike", "motorbike", "motorcycle", "van", "pickup"])
    brand_labels = set(["nissan", "toyota", "isuzu", "mitsubishi", "proton", "mazda", "mercedes", "honda", "ford", "bmw", "audi", "chevrolet"])
    vehicles = []
    plates = []
    logos = []

    for box, conf, cls in zip(boxes, scores, classes):
        label = names.get(cls, str(cls)).lower()
        item = {"box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])], "conf": round(float(conf), 3), "class": label}
        if label in plate_names or "plate" in label:
            plates.append(item)
        elif label in vehicle_name_set or label in names.values():
            vehicles.append(item)
        elif label in brand_labels or "logo" in label or any(b in label for b in ["nissan", "toyota", "honda", "mercedes", "ford", "bmw", "audi", "chevrolet", "mazda", "isuzu", "proton"]):
            logos.append(item)
        else:
            pass  # Ignore unclassified objects

    # Associate plates and logos with vehicles
    def is_inside_or_iou(parent_box, child_box):
        px1, py1, px2, py2 = child_box
        vx1, vy1, vx2, vy2 = parent_box
        pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
        if pcx >= vx1 and pcx <= vx2 and pcy >= vy1 and pcy <= vy2:
            return True
        if iou(parent_box, child_box) > IOU_ASSOC_THRESH:
            return True
        return False

    detections = []
    vehicles_with_plate = 0
    vehicles_without_plate = 0
    vehicles_with_logo = 0
    vehicles_without_logo = 0
    incoming = 0
    outgoing = 0

    mid_y = h / 2

    for v in vehicles:
        vx1, vy1, vx2, vy2 = v["box"]
        v_cx = (vx1 + vx2) / 2
        v_cy = (vy1 + vy2) / 2

        # lane detection using image width thirds
        if v_cx < w / 3:
            lane = "Left"
        elif v_cx < 2 * w / 3:
            lane = "Center"
        else:
            lane = "Right"

        # direction heuristic
        direction = "Incoming" if v_cy > mid_y else "Outgoing"

        # color for box based on direction
        color = (0, 255, 0) if direction.lower().startswith("incoming") else (0, 0, 255)

        matched_plate = next((p for p in plates if is_inside_or_iou(v["box"], p["box"])), None)
        matched_logo = next((l for l in logos if is_inside_or_iou(v["box"], l["box"])), None)

        plate_status = "Yes" if matched_plate else "No"
        logo_status = "Yes" if matched_logo else "No"
        if plate_status == "Yes":
            vehicles_with_plate += 1
        else:
            vehicles_without_plate += 1
        if logo_status == "Yes":
            vehicles_with_logo += 1
        else:
            vehicles_without_logo += 1

        # draw vehicle box
        cv2.rectangle(img, (vx1, vy1), (vx2, vy2), color, 2)

        # top label: class | direction | conf
        top_text = f"{v['class']} | {direction} | {v['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(top_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (vx1, vy1 - th - 8), (vx1 + tw + 6, vy1), (0, 0, 0), -1)
        cv2.putText(img, top_text, (vx1 + 3, vy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # bottom label: plate yes/no
        bottom_text = f"Plate: {plate_status}"
        (bw, bh), _ = cv2.getTextSize(bottom_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        color_plate = (0, 255, 255) if plate_status == "Yes" else (0, 165, 255)
        cv2.rectangle(img, (vx1, vy2 + 6), (vx1 + bw + 8, vy2 + bh + 12), color_plate, -1)
        cv2.putText(img, bottom_text, (vx1 + 3, vy2 + bh + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # draw plate box if present
        if matched_plate:
            px1, py1, px2, py2 = matched_plate["box"]
            cv2.rectangle(img, (px1, py1), (px2, py2), (0, 215, 255), 2)
            ptext = f"{matched_plate.get('class', 'plate')} {matched_plate.get('conf', 0):.2f}"
            (ptw, pth), _ = cv2.getTextSize(ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (px1, py1 - pth - 6), (px1 + ptw + 6, py1), (0, 215, 255), -1)
            cv2.putText(img, ptext, (px1 + 3, py1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # draw logo box if present
        if matched_logo:
            lx1, ly1, lx2, ly2 = matched_logo["box"]
            cv2.rectangle(img, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)  # Blue for logos
            ltext = f"{matched_logo.get('class', 'logo')}"
            (ltw, lth), _ = cv2.getTextSize(ltext, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (lx1, ly1 - lth - 6), (lx1 + ltw + 6, ly1), (255, 0, 0), -1)
            cv2.putText(img, ltext, (lx1 + 3, ly1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        detections.append({
            "class": v["class"],
            "conf": v["conf"],
            "bbox": v["box"],
            "plate_status": plate_status,
            "plate_bbox": matched_plate["box"] if matched_plate else None,
            "logo_status": logo_status,
            "logo_bbox": matched_logo["box"] if matched_logo else None,
            "lane": lane,
            "direction": direction
        })

    # header (top) summary
    total_vehicles = len(vehicles)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"Time: {timestamp} | Vehicles: {total_vehicles} | With Plates: {vehicles_with_plate} | Without Plates: {vehicles_without_plate} | With Logos: {vehicles_with_logo} | Without Logos: {vehicles_without_logo} | Incoming: {incoming} | Outgoing: {outgoing}"
    cv2.rectangle(img, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(img, header, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # save annotated
    out_path = os.path.join(OUTPUT_DIR, f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    cv2.imwrite(out_path, img)

    summary = {
        "timestamp": timestamp,
        "num_detections": total_vehicles,
        "vehicle_count": total_vehicles,
        "vehicles_with_plate": vehicles_with_plate,
        "vehicles_without_plate": vehicles_without_plate,
        "vehicles_with_logo": vehicles_with_logo,
        "vehicles_without_logo": vehicles_without_logo,
        "incoming": incoming,
        "outgoing": outgoing,
        "detections": detections
    }

    result = {"summary": summary, "annotated_image_path": out_path}
    if return_b64:
        with open(out_path, "rb") as f:
            result["annotated_image_base64"] = base64.b64encode(f.read()).decode("utf-8")
    return result