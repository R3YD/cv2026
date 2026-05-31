import cv2
import numpy as np
import json

def load_calibration_from_json(json_str):
    data = json.loads(json_str)
    cam_info = data["cameras"][0]
    mtx = np.array(cam_info["camera_matrix"])
    dist = np.array(cam_info["distortion"])
    return mtx, dist

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def calculate_pov_degree(rect):
    """
    Рассчитывает угол наклона по X (Pitch) и Y (Yaw) 
    на основе перспективного сокращения сторон.
    """
    top = np.linalg.norm(rect[1] - rect[0])
    bottom = np.linalg.norm(rect[2] - rect[3])
    left = np.linalg.norm(rect[3] - rect[0])
    right = np.linalg.norm(rect[2] - rect[1])

    if max(top, bottom) == 0 or max(left, right) == 0:
        return 0.0, 0.0

    ratio_x = min(top, bottom) / max(top, bottom)
    ratio_y = min(left, right) / max(left, right)

    degree_x = np.degrees(np.arccos(ratio_x))
    degree_y = np.degrees(np.arccos(ratio_y))

    return degree_x, degree_y

def correct_perspective(frame, rect):
    (tl, tr, br, bl) = rect
    
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    
    if maxWidth < 10 or maxHeight < 10:
        return None

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)
    return warped

def main():
    calib_json = r'''{"cameras": [{"id": "ca768b07-1e75-4d8b-aaae-6e64cfaec250", "type": "rgb", "calibration_source": 0,
    "camera_matrix": [[636.3095298640643, 0.0, 319.5], [0.0, 642.3181334658376, 239.5], [0.0, 0.0, 1.0]],
    "optimal_camera_matrix": [[581.4934495063519, 0.0, 306.78948888762244], [0.0, 601.5997089946579, 254.68260524302357], [0.0, 0.0, 1.0]],
    "roi": [22, 34, 576, 434],
    "distortion": [[0.14470478425308592, 1.7775207856507023, 0.016626932343971453, -0.004810534251972255, -8.672086016421774]]}], 
    "board": {"type": "chess", "pattern_size": [9, 6], "square_size": 0.025}}'''
    mtx, dist = load_calibration_from_json(calib_json)

    cap = cv2.VideoCapture(3)         
    detector = cv2.QRCodeDetector()

    mode = "raw"                 
    use_calibration = False

    # Раздельный учет углов
    max_raw_x, max_raw_y = 0.0, 0.0
    max_corr_x, max_corr_y = 0.0, 0.0

    print("Управление: r - RAW | c - CORRECTED | k - Калибровка | q - Выход")
    cv2.namedWindow("QR Scanner", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret: break

        if use_calibration:
            frame = cv2.undistort(frame, mtx, dist, None, mtx)

        ret_detect, bbox = detector.detect(frame)
        
        info = f"Mode: {mode.upper()} | Calib: {'ON' if use_calibration else 'OFF'}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        if ret_detect and bbox is not None:
            rect = order_points(bbox[0])
            deg_x, deg_y = calculate_pov_degree(rect)
            
            data = ""
            
            if mode == "raw":
                data, _ = detector.decode(frame, bbox)
                if data:
                    max_raw_x = max(max_raw_x, deg_x)
                    max_raw_y = max(max_raw_y, deg_y)
                
                for i in range(4):
                    pt1 = tuple(rect[i].astype(int))
                    pt2 = tuple(rect[(i+1)%4].astype(int))
                    cv2.line(frame, pt1, pt2, (0,0,255), 2)

            elif mode == "corrected":
                warped = correct_perspective(frame, rect)
                
                if warped is not None:
                    warped_padded = cv2.copyMakeBorder(warped, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    
                    data, _, _ = detector.detectAndDecode(warped_padded)
                    
                    if data:
                        max_corr_x = max(max_corr_x, deg_x)
                        max_corr_y = max(max_corr_y, deg_y)

                    th, tw = 150, 150
                    small = cv2.resize(warped_padded, (tw, th))
                    frame[0:th, frame.shape[1]-tw:frame.shape[1]] = small
                
                for i in range(4):
                    pt1 = tuple(rect[i].astype(int))
                    pt2 = tuple(rect[(i+1)%4].astype(int))
                    cv2.line(frame, pt1, pt2, (0,255,0), 2)

            cv2.putText(frame, f"Tilt X: {deg_x:.1f} | Y: {deg_y:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            if data:
                cv2.putText(frame, f"Data: {data}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("QR Scanner", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('й'), 27]: break
        elif key in [ord('r'), ord('к')]: mode = "raw"
        elif key in [ord('c'), ord('с')]: mode = "corrected"
        elif key in [ord('k'), ord('л')]: use_calibration = not use_calibration

    print("\n=== РЕЗУЛЬТАТЫ (Максимальные углы) ===")
    print(f"RAW Mode       -> X: {max_raw_x:.1f}°, Y: {max_raw_y:.1f}°")
    print(f"CORRECTED Mode -> X: {max_corr_x:.1f}°, Y: {max_corr_y:.1f}°")
    print("======================================\n")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
