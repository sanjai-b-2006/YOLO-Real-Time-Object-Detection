import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import mimetypes
from PIL import Image, ImageSequence
# Load YOLO
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Object Detection on Image or Frame
def detect_objects_image(image):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 1.8)
                y = int(center_y - h / 1.8)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{label} {round(confidence, 2)}", (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return image

# App UI
st.set_page_config(layout="wide")
st.markdown("<h1 style='font-size: 48px;'>Object Detection Using YOLO</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi","webp","wmv","gif"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    if "image" in file_type and not uploaded_file.name.endswith(".gif"):
        image = cv2.imread(tfile.name)
        image = detect_objects_image(image)

        # Resize output by 1.5x
        image = cv2.resize(image, None, fx=1.5, fy=1.5)
        st.image(image, channels="BGR", caption="Detected Image")
    
    elif "video" in file_type:
        stframe = st.empty()
        cap = cv2.VideoCapture(tfile.name)
        frame_id = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            frame = detect_objects_image(frame)

            # Show FPS
            elapsed_time = time.time() - start_time
            fps = frame_id / elapsed_time
            cv2.putText(frame, f"FPS: {round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            frame = cv2.resize(frame, None, fx=1.5, fy=1.5)
            stframe.image(frame, channels="BGR")

        cap.release()
    elif uploaded_file.name.endswith(".gif"):
        gif = Image.open(tfile.name)
        stframe = st.empty()

        for frame in ImageSequence.Iterator(gif):
            frame_rgb = frame.convert("RGB")
            open_cv_image = np.array(frame_rgb)
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

            open_cv_image = detect_objects_image(open_cv_image)
            open_cv_image = cv2.resize(open_cv_image, None, fx=1.5, fy=1.5)
            stframe.image(open_cv_image, channels="BGR")

    else:
        st.error("Unsupported file format.")
