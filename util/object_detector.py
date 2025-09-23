"""
Object detector – now with YOLOv8 fallback.

detect_object(bgr, use_yolo=True) →  (found, cx, cy, r)
object_is_captured(...)           →  True/False
visualize_object_detection(...)   →  OpenCV overlay (optional)
"""
import cv2, numpy as np
from ultralytics import YOLO   #  <-- requires  `pip install ultralytics`

# ------------------------------------------------------------------ #
# YOLOv8 INITIALISATION (loaded once, cached globally)  ------------- #
# ------------------------------------------------------------------ #
_yolo = None
def _load():
    global _yolo
    if _yolo is None:
        _yolo = YOLO("yolov8n.pt")      # COCO-pre-trained (class-32 = sports-ball)
        _yolo.fuse()                   # +10-15 % speed at inference
    return _yolo

# ------------------------------------------------------------------ #
# YOLOv8 INFERENCE  ------------------------------------------------- #
# ------------------------------------------------------------------ #
def _yolo_detect(bgr, conf=0.25):
    """
    YOLOv8 single-class detector.

    Returns
    -------
    found : bool
    cx, cy: int      - centre pixel of the *largest* detected ball
    r      : int      - half-diagonal of its bounding box
    bbox   : tuple    - (x1, y1, x2, y2)  top-left & bottom-right ints
    """
    model = _load()
    preds = model(bgr, imgsz=640, conf=conf, iou=0.4, verbose=False)[0]

    # COCO “sports ball” → class-id 32  (change if you train a custom class)
    keep = preds.boxes.cls.cpu().numpy() == 32
    # COCO bicycle” → class-id 1  (change if you train a custom class)
    # keep = preds.boxes.cls.cpu().numpy() == 1
    if not np.any(keep):
        return False, None, None, None, None

    boxes  = preds.boxes.xyxy.cpu().numpy()[keep]            # x1 y1 x2 y2
    areas  = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    x1, y1, x2, y2 = boxes[np.argmax(areas)]
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    r      = int(0.5 * np.linalg.norm([x2 - x1, y2 - y1]))
    return True, cx, cy, r, (int(x1), int(y1), int(x2), int(y2))

# ------------------------------------------------------------------ #
# PUBLIC API  ------------------------------------------------------- #
# ------------------------------------------------------------------ #
def detect_object(bgr, use_yolo=True):
    if use_yolo:
        return _yolo_detect(bgr)
    raise NotImplementedError("HSV mode removed – always use YOLO now.")

def object_is_captured(found, cx, cy, r,
                     frame_w, frame_h,
                     # horizontal window: centre ± 10 % of width
                     centre_frac_x=0.10,
                     # vertical window: lower half acceptable – use 30 % of height
                     centre_frac_y=0.30,
                     # how “big” (close) the ball must appear: ≥ 12 % of frame height
                     min_radius_frac=0.12):
    """
    True  ⇢ the ball is ‘captured’  ⤳  big *and* centred (within given windows)

    • centre_frac_x  – relative half-width of horizontal capture zone
    • centre_frac_y  – relative half-height of vertical   capture zone
      (set larger if you’re happy with the ball being lower in the image)
    • min_radius_frac– ball must appear at least this big (proxy for distance)
    """
    if not found:
        return False

    centre_x = frame_w / 2
    centre_y = frame_h / 2

    x_ok = abs(cx - centre_x) <= (centre_frac_x * frame_w)
    # allow lower-part bias ⇒ compare to *lower* third of image:
    y_ok = (cy >= centre_y - centre_frac_y * frame_h) and \
           (cy <= centre_y + centre_frac_y * frame_h)

    r_ok = r >= min_radius_frac * frame_h
    return x_ok and y_ok and r_ok

def visualize_object_detection(frame, found, cx, cy, r, bbox, win="ball"):
    """
    Draw a cyan rectangle plus red cross for the detected ball.
    bbox should be (x1,y1,x2,y2) ints – pass straight from _yolo_detect.
    """
    if not found or bbox is None:
        return
    x1, y1, x2, y2 = bbox
    disp = frame.copy()

    # cyan rectangle
    cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 255, 0), 2)
    # red cross at centre
    cv2.drawMarker(disp, (cx, cy), (0, 0, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

    cv2.imshow(win, disp)
    cv2.waitKey(1)