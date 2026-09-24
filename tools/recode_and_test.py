import cv2
import torch
import sys
import time
sys.path.append('.')

from yolox.exp import get_exp
from yolox.data.data_augment import preproc
from yolox.utils import postprocess

# Step 1: 重编码视频
print("Step 1: Recoding video...")
input_path = 'video/walker1.mp4'
output_path = 'video/walker1_recoded.mp4'

cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

count = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    out.write(frame)
    count += 1

cap.release()
out.release()
print(f"  Recoded {count} frames -> {output_path}")

# Step 2: 检测每帧目标数量
print("\nStep 2: Counting detections per frame...")
exp = get_exp('yolox/exps/example/mot/yolox_s_mix_det.py', None)
model = exp.get_model().to('cpu')
ckpt = torch.load('pretrained/bytetrack_s_mot17.pth.tar', map_location='cpu')
model.load_state_dict(ckpt["model"])
model.eval()

cap = cv2.VideoCapture(output_path)
det_counts = []
for i in range(100):
    ret, frame = cap.read()
    if not ret: break

    img, _ = preproc(frame, exp.test_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    img = torch.from_numpy(img).unsqueeze(0).float()

    with torch.no_grad():
        outputs = model(img)
        outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)

    if outputs[0] is not None:
        det_counts.append(len(outputs[0]))
    else:
        det_counts.append(0)

cap.release()

print(f"  Frames processed: {len(det_counts)}")
print(f"  Avg detections/frame: {sum(det_counts)/len(det_counts):.1f}")
print(f"  Max: {max(det_counts)}, Min: {min(det_counts)}")
print(f"  Frames with detections: {sum(1 for d in det_counts if d > 0)}")

print("\nDone. Now run:")
print(f"python tools/demo.py video --path {output_path} -f yolox/exps/example/mot/yolox_s_mix_det.py -c pretrained/bytetrack_s_mot17.pth.tar --fp16 --fuse --save_result")
