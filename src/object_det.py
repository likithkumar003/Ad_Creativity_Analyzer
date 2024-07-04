# ==================================================================================================================



import argparse
from ultralytics import YOLO



# ==================================================================================================================




# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('-img_path', '--i', '-I', type=str, nargs=1, help='Location of the image to be scanned')
parser.add_argument('-dev', '--d', '-D', type=int, choices=[0, 1, 2], help='Device usage: 0 for CPU, 1 for CUDA, 2 for multi-GPU (not supported yet)')
args = parser.parse_args()

names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}



# ==================================================================================================================




def run_det(img_path: str, dev: int) -> list:
    try:
        model = YOLO(r"models/yolov8m.pt")
        if dev == 0:
            results = model(img_path, device="cpu", verbose=True)
        elif dev == 1:
            results = model(img_path, device="cuda", verbose=True, save_dir=r"E:")
        output = []
        for i in results:
            boxes = i.boxes
            for j in boxes.cls:
                op = str(j).split("(")[1].split(".")[0]
                output.append(names[int(op)])
        return output
    except Exception as e:
        print("Encountered error:", e)



# ==================================================================================================================




if args.i and args.d is not None:
    run_det(args.i, args.d)



