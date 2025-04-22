import cv2
import numpy as np
import asyncio
import httpx
import time

# Load model
net = cv2.dnn.readNetFromDarknet("model/merge.cfg", "model/merge_yolov4.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
layer_names = net.getUnconnectedOutLayersNames()

conf_threshold = 0.5
nms_threshold = 0.4
cooldown_secs = 15
output_path = "data/image.jpeg"
server_url = "http://127.0.0.1:8123/analyze_image/"

async def send_image(image_path):
    try:
        with open(image_path, "rb") as f:
            files = {"file": ("image.jpeg", f, "image/jpeg")}
            async with httpx.AsyncClient() as client:
                response = await client.post(server_url, files=files)
        print("Sent to server. Status code:", response.status_code)
    except Exception as e:
        print("Error sending image:", e)

def detect_objects(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    boxes, confidences, class_ids = [], [], []

    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x, center_y, width, height = (detection[0:4] * [w, h, w, h]).astype(int)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return indices, boxes, confidences, class_ids

async def main_loop():
    cap = cv2.VideoCapture(1)
    last_send_time = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Camera read failed")
                break

            indices, boxes, confidences, class_ids = detect_objects(frame)
            now = time.time()
            detection_occurred = len(indices) > 0

            if detection_occurred:
                last_detection = time.time()
                for i in indices:
                    i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
                    x, y, w_box, h_box = boxes[i]
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_ids[i]} {confidences[i]:.2f}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if now - last_send_time > cooldown_secs:
                    cv2.imwrite(output_path, frame)
                    asyncio.create_task(send_image(output_path))
                    last_send_time = now
            cv2.imshow("Live Detection", frame)

            await asyncio.sleep(0.03)
            # Show window regardless of detection
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main_loop())
