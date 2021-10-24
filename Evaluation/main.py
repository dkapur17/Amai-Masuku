from imutils.video import FPS
from centroid_tracker import CentroidTracker
import numpy as np
import argparse
import cv2
import tqdm


def write_to_csv(csv_file, frame_no, csv_obj):
    try:
        with open(csv_file) as f:
            pass
    except FileNotFoundError:
        with open(csv_file, 'x') as f:
            f.write(
                "Frame Number, Total non-masked faces, Total masked faces, Non-masked Face ROIs, Masked Faces ROIs\n")

    line = str(frame_no) + ','
    non_mask_roi = []
    mask_roi = []
    with_mask = 0

    for obj in csv_obj:
        (x, y, w, h, mask) = obj
        if mask == 0:
            mask_roi.append(f"{x},{y},{w},{h}")
            with_mask += 1
        else:
            non_mask_roi.append(f"{x},{y},{w},{h}")
    line += str(len(csv_obj) - with_mask) + ',' + \
        str(with_mask) + ',' + ';'.join(non_mask_roi) + ',' + ';'.join(mask_roi)
    with open(csv_file, 'a') as f:
        f.write(line + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, default="",
                    help="path to (optional) input video file")
    ap.add_argument("-o", "--output", type=str, default="",
                    help="path to (optional) output video file")
    ap.add_argument("-d", "--display", type=int, default=1,
                    help="whether or not output frame should be displayed")
    ap.add_argument("-c", "--confidence", type=float, default=0.45,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
                    help="threshold when applyong non-maxima suppression")
    ap.add_argument("-C", "--csv", type=str, required=True,
                    help="CSV file to output frame statistics")
    ap.add_argument("-T", "--txt", type=str, required=True,
                    help="Text file to output subject statistics")

    args = vars(ap.parse_args())

    LABELS = ['No Mask', 'Mask']
    COLORS = [[0, 0, 255], [0, 255, 0]]
    ct = CentroidTracker()

    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet("mask.cfg", "mask.weights")

    ln = net.getLayerNames()
    try:
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    W = None
    H = None

    print("[INFO] accessing video stream...")
    vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
    writer = None
    fps = FPS().start()
    frame_count = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_no in tqdm.tqdm(range(frame_count)):
        (ret, frame) = vs.read()

        if not ret:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (864, 864), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes = []
        csv_obj = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > args["confidence"]:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(
            boxes, confidences, args["confidence"], args["threshold"])

        filtered_boxes = []
        filtered_classIDs = []

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                filtered_boxes.append((x, y, w, h))
                filtered_classIDs.append(classIDs[i])
                csv_obj.append((x, y, w, h, classIDs[i]))

        objects = ct.update(filtered_boxes, filtered_classIDs)
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = f"ID {objectID}"
            cv2.putText(frame, text, (centroid[0] + 20, centroid[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # write to csv file
        write_to_csv(args['csv'], frame_no, csv_obj)

        if args["display"] > 0:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        if args["output"] != "" and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                args["output"], fourcc, 24, (frame.shape[1], frame.shape[0]), True)

        if writer is not None:
            writer.write(frame)

        fps.update()
    ct.write_file(args['txt'])
    fps.stop()
    cv2.destroyAllWindows()
