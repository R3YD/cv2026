import cv2
import numpy as np
import math
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

def correct_perspective(frame, rect):
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    size = 300
    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (size, size))
    return warped

def estimate_angle(rect):
    (tl, tr, _, _) = rect
    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]
    return math.degrees(math.atan2(dy, dx))

def smooth_bbox(prev, current, alpha=0.7):
    if prev is None:
        return current
    return alpha * prev + (1 - alpha) * current

def main():
    # Полная калибровочная строка (валидный JSON)
    calib_json = r'''{"cameras": [{"id": "ca768b07-1e75-4d8b-aaae-6e64cfaec250", "type": "rgb", "calibration_source": 0,
    "camera_matrix": [[636.3095298640643, 0.0, 319.5], [0.0, 642.3181334658376, 239.5], [0.0, 0.0, 1.0]],
    "optimal_camera_matrix": [[581.4934495063519, 0.0, 306.78948888762244], [0.0, 601.5997089946579, 254.68260524302357], [0.0, 0.0, 1.0]],
    "roi": [22, 34, 576, 434],
    "distortion": [[0.14470478425308592, 1.7775207856507023, 0.016626932343971453, -0.004810534251972255, -8.672086016421774]],
    "rvecs": [[[0.16758400583486135],[0.015533411689779973],[-0.021403875548799522]], [[0.26368904127184556],[-0.012073283560185992],[1.5819517342598055]], [[0.15354388779370584],[-0.4170413916780201],[3.0874763337557565]], [[0.044591803443880515],[0.30636396320260145],[-1.5141982381014198]], [[0.215651231237642],[-0.007052656665205688],[-0.20115407426882595]], [[0.2050135408277052],[-0.17699541031950208],[0.27833894316121544]], [[0.21630184193161778],[0.007498640986453781],[1.8156665952734865]], [[0.31505109228810885],[0.052512572552245756],[1.3630893545900562]], [[-0.010731903739002487],[0.20180116158549005],[-2.8904245064718355]], [[0.36348475133080627],[-0.24074282057857752],[2.7974327467145783]], [[-0.06411052687818267],[0.16251662206389683],[-1.4271182569219854]], [[0.04094833810483448],[0.18874097320969602],[-1.429979291557418]], [[-0.11889365537181111],[0.0893529161469573],[-0.10801451190317793]], [[0.3670554793088149],[-0.03572339443185511],[-0.047336957834097554]], [[0.2398382907331238],[0.2800223122973557],[0.08957774141340126]], [[0.054007447865303024],[-0.42490426862000846],[-0.1616458052273562]], [[-0.3262139891402814],[-0.02441730783618817],[-0.008722294792369816]], [[0.40496900072185754],[0.04094021365421842],[-0.11278340061967018]], [[-0.012286082706865424],[-0.45133177558327925],[2.5945400935329404]], [[0.08237673551308065],[0.24163774885197695],[-2.6001808034974396]], [[0.3104147247396937],[-0.06814736332985528],[-0.5893492695455368]], [[0.24304829312835835],[-0.15439794359619105],[1.5107200864921584]], [[0.05413726173994428],[0.07573260269063888],[1.6203531965196738]], [[0.21165043327946476],[0.14725408103997234],[1.9972970567135855]], [[0.16426863055046131],[-0.08187830024217371],[3.1064552160585834]], [[-0.11528338463084889],[-0.029908070011886796],[3.0957111406396773]], [[0.05572376494431078],[0.03837917711462577],[-2.9151033455614828]], [[0.17176753977569303],[-0.21603332428125332],[2.831509256392598]], [[0.04482028515361345],[0.03094450826027117],[0.027708394909232163]], [[0.18364534256388018],[0.10605397017637393],[0.3524235336758166]], [[0.1349667233603786],[0.19783890931863515],[-0.24478270028788188]]],
    "tvecs": [[[-0.09931764711610076],[-0.05344970004343794],[0.33844433037115434]], [[0.06648827890310258],[-0.10562721564563836],[0.3403781672062564]], [[0.11001437787624238],[0.05685221555590893],[0.33550572085816543]], [[-0.058470147676408],[0.09238519312154438],[0.3931464368283444]], [[-0.10839799339289732],[-0.03146069127063157],[0.32135318195892537]], [[-0.05542305684781787],[-0.07839151206559204],[0.2875439022680785]], [[0.09776637003980711],[-0.08312106701691312],[0.3496385155944161]], [[0.06585493849422391],[-0.11196999318877879],[0.32947809870994516]], [[0.10156766042292587],[0.07283501034119369],[0.33682189548732455]], [[0.11303666210979813],[0.03053657661493229],[0.3019730125628852]], [[-0.029013813318488676],[0.08420210363735799],[0.4032130620221558]], [[-0.025743481770868226],[0.08149190274120659],[0.4139255457887266]], [[-0.0948331117582572],[-0.05353978491196403],[0.3891906075383187]], [[-0.10473878387611435],[-0.037486879410851694],[0.33744266608653367]], [[-0.08623811502419734],[-0.07790390135678908],[0.38270376295857333]], [[-0.07719515830856839],[-0.047105975660229296],[0.2846112701156524]], [[-0.09485153113157052],[-0.08068963036909493],[0.36807888969997454]], [[-0.10905584932419325],[-0.040718543513000506],[0.3439536182413544]], [[0.13836072241634587],[0.0002653561757810076],[0.3791917091898237]], [[0.08442105180600945],[0.10113900903355846],[0.3907706546543804]], [[-0.10231104665622694],[0.0071719357996141465],[0.3515934390054946]], [[0.06055903320176899],[-0.10230378505837524],[0.34592003703415247]], [[0.059773290135497335],[-0.10126467976168604],[0.4291968013777585]], [[0.13051470029695933],[-0.0735928701540763],[0.3898664641511442]], [[0.12980733729373534],[0.036311357842941115],[0.3954441722247202]], [[0.1286938836873859],[0.040353411248279084],[0.3933642423320933]], [[0.11044622975492799],[0.06112948108068581],[0.39726049246361167]], [[0.12702996740139078],[0.025110796871907162],[0.3750978309507272]], [[-0.07940346336517642],[-0.073071204704716],[0.39851233102673644]], [[-0.06008999052281762],[-0.09574449481226657],[0.39590449593193394]], [[-0.10124147838022206],[-0.027996034299388968],[0.39475905548722856]]],
    "resolution": {"h": 480, "w": 640}, "total_error": 0.1887256326159488}], "board": {"type": "chess", "pattern_size": [9, 6], "square_size": 0.025}}'''
    mtx, dist = load_calibration_from_json(calib_json)

    cap = cv2.VideoCapture(3)          # индекс вашей DroidCam
    if not cap.isOpened():
        print("Не удалось открыть камеру 3")
        return

    detector = cv2.QRCodeDetector()

    mode = "corrected"                 # "raw" или "corrected"
    use_calibration = False
    prev_bbox = None
    max_angle_raw = 0.0
    max_angle_corrected = 0.0

    print("Управление:")
    print("  r - RAW      c - CORRECTED")
    print("  k - калибровка вкл/выкл")
    print("  q / ESC - выход")
    print("Кликните на окно 'QR Scanner', чтобы клавиши работали.")

    cv2.namedWindow("QR Scanner", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if use_calibration:
            frame = cv2.undistort(frame, mtx, dist, None, mtx)

        data, bbox, _ = detector.detectAndDecode(frame)

        # Инфо-строка
        info = f"Mode: {mode.upper()} | Calib: {'ON' if use_calibration else 'OFF'}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        current_angle = None

        if bbox is not None:
            bbox = bbox.astype(np.float32)
            if mode == "corrected":
                bbox = smooth_bbox(prev_bbox, bbox)
                prev_bbox = bbox
                rect = order_points(bbox[0])
                for i in range(4):
                    pt1 = tuple(rect[i].astype(int))
                    pt2 = tuple(rect[(i+1)%4].astype(int))
                    cv2.line(frame, pt1, pt2, (255,0,0), 2)

                # Выпрямляем QR и вставляем в левый верхний угол
                corrected = correct_perspective(frame, rect)
                h, w = frame.shape[:2]
                th, tw = h // 4, w // 4
                if corrected is not None:
                    small = cv2.resize(corrected, (tw, th))
                    frame[0:th, 0:tw] = small

                current_angle = estimate_angle(rect)
                if current_angle is not None:
                    max_angle_corrected = max(max_angle_corrected, abs(current_angle))
            else:  # raw
                pts = bbox[0].astype(int)
                for i in range(4):
                    cv2.line(frame, tuple(pts[i]), tuple(pts[(i+1)%4]), (255,0,0), 2)
                pt1, pt2 = bbox[0][0], bbox[0][1]
                dx = pt2[0] - pt1[0]
                dy = pt2[1] - pt1[1]
                current_angle = np.degrees(np.arctan2(dy, dx))
                max_angle_raw = max(max_angle_raw, abs(current_angle))

            if current_angle is not None:
                cv2.putText(frame, f"Angle: {current_angle:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            if data:
                cv2.putText(frame, f"Data: {data}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
            prev_bbox = None

        cv2.imshow("QR Scanner", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            mode = "raw"
            print("Режим переключён на RAW")
        elif key == ord('c'):
            mode = "corrected"
            print("Режим переключён на CORRECTED")
        elif key == ord('k'):
            use_calibration = not use_calibration
            print(f"Калибровка {'включена' if use_calibration else 'выключена'}")

    # Сохранение результатов
    with open("angle_results.txt", "w") as f:
        f.write(f"Max angle RAW: {max_angle_raw:.2f}\n")
        f.write(f"Max angle CORRECTED: {max_angle_corrected:.2f}\n")
    print(f"Max angle RAW: {max_angle_raw:.2f}")
    print(f"Max angle CORRECTED: {max_angle_corrected:.2f}")
    print("Результаты сохранены в angle_results.txt")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()