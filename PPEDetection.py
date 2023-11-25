from ultralytics import YOLO
import cvzone
import cv2 as cv
import math

# cap = cv2.VideoCapture(0) # For webcam
# cap.set(3, 640)
# cap.set(4, 480)

cap = cv.VideoCapture("../Videos/ppe-2.mp4")

# Creating a video from the resulting detection
fps = cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fourcc = cv.VideoWriter.fourcc(*'XVID')

cap_out = cv.VideoWriter('output2.avi', fourcc, fps, (width, height))
model = YOLO("ppe.pt")

classNames = ['hardhat', 'mask', 'no-hardhat', 'no-mask', 'no-safety vest',
              'person', 'safety cone', 'safety vest', 'machinery', 'vehicle']

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # We can check for individual bounding boxes to see how well it performs
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            bbox = int(x1), int(y1), int(w), int(h)

            # For the confidence level
            conf = math.ceil((box.conf[0] * 100)) / 100
            print(conf)

            # Class names
            cls = int(box.cls[0])
            cclass = classNames[cls]

            # Check for the good classes
            if conf > 0.3:
                if cclass == "hardhat" or cclass == "mask" or cclass == "safety vest":
                    cv.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=3)
                    cvzone.putTextRect(img, f"{cclass} {conf}", (max(0, x1), max(35, y1)),
                                       scale=0.5, thickness=2, colorB=(0, 255, 0), colorT=(0, 0, 0),
                                       colorR=(0, 255, 0), offset=5, font=cv.FONT_HERSHEY_SIMPLEX)
                elif cclass == "person":
                    cv.rectangle(img, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=3)
                    cvzone.putTextRect(img, f"{cclass} {conf}", (max(0, x1), max(35, y1)),
                                       scale=0.5, thickness=2, colorB=(255, 255, 255), colorT=(0, 0, 0),
                                       colorR=(255, 255, 255),
                                       offset=5, font=cv.FONT_HERSHEY_SIMPLEX)

                else:
                    cv.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)
                    cvzone.putTextRect(img, f"{cclass} {conf}", (max(0, x1), max(35, y1)),
                                       scale=0.5, thickness=2, colorB=(0, 0, 255), colorT=(0, 0, 0),
                                       colorR=(0, 0, 255), offset=5, font=cv.FONT_HERSHEY_SIMPLEX)

    cap_out.write(img)
    cv.imshow("image", img)
    cv.waitKey(1)
