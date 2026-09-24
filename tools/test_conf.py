import cv2
import torch
import sys
import time
sys.path.append('.')

from yolox.exp import get_exp
from yolox.data.data_augment import preproc
from yolox.utils import postprocess

video_path = 'video/前保_30s.mp4'
conf_threshold = 0.5

print(f"Testing video: {video_path}")
print(f"Confidence threshold: {conf_threshold}")
print()

exp = get_exp('yolox/exps/example/mot/yolox_s_mix_det.py', None)
exp.test_conf = conf_threshold

model = exp.get_model().to('cpu')
ckpt = torch.load('pretrained/bytetrack_s_mot17.pth.tar', map_location='cpu')
model.load_state_dict(ckpt["model"])
model.eval()

cap = cv2.VideoCapture(video_path)
det_counts = []
times = []

for i in range(50):
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    img, _ = preproc(frame, exp.test_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    img = torch.from_numpy(img).unsqueeze(0).float()
    with torch.no_grad():
        outputs = model(img)
        outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)
    times.append(time.time() - start)

    if outputs[0] is not None:
        det_counts.append(len(outputs[0]))
    else:
        det_counts.append(0)

cap.release()

print(f"Frames processed: {len(det_counts)}")
print(f"Avg detections/frame: {sum(det_counts)/len(det_counts):.1f}")
print(f"Max detections: {max(det_counts)}")
print(f"Avg inference time: {sum(times)/len(times)*1000:.1f}ms")
print(f"FPS (detection only): {1/(sum(times)/len(times)):.1f}")
