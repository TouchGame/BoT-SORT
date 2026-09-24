import cv2
import torch
import sys
import time
sys.path.append('.')

from yolox.exp import get_exp
from yolox.data.data_augment import preproc
from yolox.utils import postprocess, fuse_model
from tracker.bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer

video_path = 'video/前保_30s.mp4'
conf_threshold = 0.5

print(f'Testing with conf={conf_threshold}')
print()

exp = get_exp('yolox/exps/example/mot/yolox_s_mix_det.py', None)
exp.test_conf = conf_threshold

device = torch.device('cuda')
model = exp.get_model().to(device)
ckpt = torch.load('pretrained/bytetrack_s_mot17.pth.tar', map_location='cpu')
model.load_state_dict(ckpt["model"])
model = fuse_model(model)
model = model.half()
model.eval()

class Args:
    track_high_thresh = 0.6
    track_low_thresh = 0.1
    new_track_thresh = 0.7
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = 1.6
    min_box_area = 100
    fuse_score = False
    cmc_method = 'orb'
    with_reid = False
    proximity_thresh = 0.5
    appearance_thresh = 0.25
    name = 'benchmark'
    ablation = False
    mot20 = False

cap = cv2.VideoCapture(video_path)
timer_det = Timer()
timer_track = Timer()
tracker = BoTSORT(args=Args(), frame_rate=30)

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detection
    timer_det.tic()
    img, ratio = preproc(frame, exp.test_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    img = torch.from_numpy(img).unsqueeze(0).to(device).half()
    with torch.no_grad():
        outputs = model(img)
        outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)
    timer_det.toc()

    # Tracking
    timer_track.tic()
    if outputs[0] is not None:
        det = outputs[0].cpu().numpy()
        det = det[:, :7]
        det[:, :4] /= ratio
        online_targets = tracker.update(det, frame)
    timer_track.toc()

    if frame_id % 10 == 0:
        det_fps = 1 / timer_det.average_time if timer_det.average_time > 0 else 0
        track_fps = 1 / timer_track.average_time if timer_track.average_time > 0 else 0
        print(f'Frame {frame_id}: det={det_fps:.1f} fps, track={track_fps:.1f} fps')

    frame_id += 1

cap.release()
print()
print(f'Total frames: {frame_id}')
