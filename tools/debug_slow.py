import sys
import argparse
import os
import os.path as osp
import time
import cv2
import torch
import numpy as np

from loguru import logger

sys.path.append('.')

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from tracker.bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer

def make_parser():
    parser = argparse.ArgumentParser("BoT-SORT Debug")
    parser.add_argument("demo", default="video", help="demo type")
    parser.add_argument("-f", "--exp_file", default=None, type=str)
    parser.add_argument("-c", "--ckpt", default=None, type=str)
    parser.add_argument("--path", default="", help="path to video")
    parser.add_argument("--conf", default=None, type=float)
    parser.add_argument("--tsize", default=None, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true")
    parser.add_argument("--save_result", action="store_true")
    parser.add_argument("--track_high_thresh", type=float, default=0.6)
    parser.add_argument("--track_low_thresh", default=0.1, type=float)
    parser.add_argument("--new_track_thresh", default=0.7, type=float)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=0.8)
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6)
    parser.add_argument('--min_box_area', type=float, default=10)
    parser.add_argument("--fuse-score", dest="fuse_score", default=False, action="store_true")
    parser.add_argument("--cmc-method", default="orb", type=str)
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth")
    parser.add_argument('--proximity_thresh', type=float, default=0.5)
    parser.add_argument('--appearance_thresh', type=float, default=0.25)
    return parser


class Predictor(object):
    def __init__(self, model, exp, device, fp16):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()

        timer.tic()
        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_info


def main(exp, args):
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    model.eval()

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])

    if args.fuse:
        model = fuse_model(model)
    if args.fp16:
        model = model.half()

    predictor = Predictor(model, exp, args.device, args.fp16)

    cap = cv2.VideoCapture(args.path)
    tracker = BoTSORT(args, frame_rate=args.fps)
    timer_total = Timer()
    timer_det = Timer()
    timer_gmc = Timer()
    timer_match = Timer()
    timer_kalman = Timer()

    frame_id = 0
    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            break

        timer_total.tic()

        # Detection
        timer_det.tic()
        outputs, img_info = predictor.inference(frame, timer_det)
        timer_det.toc()

        if outputs[0] is not None:
            outputs = outputs[0].cpu().numpy()
            detections = outputs[:, :7]
            scale = min(exp.test_size[0] / float(img_info['height']), exp.test_size[1] / float(img_info['width']))
            detections[:, :4] /= scale
            dets = detections[:, :4]
            scores = detections[:, 4]
            classes = detections[:, 6]
        else:
            dets = []
            scores = []
            classes = []

        # Time each part of tracker.update
        t0 = time.time()
        timer_kalman.tic()
        # Simulate Kalman prediction
        if len(tracker.tracked_stracks) > 0:
            from tracker.basetrack import TrackState
            strack_pool = tracker.tracked_stracks + tracker.lost_stracks
            STrack.multi_predict(strack_pool)
        timer_kalman.toc()

        t1 = time.time()
        timer_gmc.tic()
        warp = tracker.gmc.apply(frame, dets if len(dets) > 0 else None)
        timer_gmc.toc()

        t2 = time.time()
        timer_match.tic()
        # Run the full update to measure matching time
        online_targets = tracker.update(detections, frame)
        timer_match.toc()

        t3 = time.time()
        timer_total.toc()

        if frame_id % 20 == 0:
            total_fps = 1 / timer_total.average_time if timer_total.average_time > 0 else 0
            det_fps = 1 / timer_det.average_time if timer_det.average_time > 0 else 0
            gmc_fps = 1 / timer_gmc.average_time if timer_gmc.average_time > 0 else 0
            match_fps = 1 / timer_match.average_time if timer_match.average_time > 0 else 0
            kalman_fps = 1 / timer_kalman.average_time if timer_kalman.average_time > 0 else 0
            tracked = len(tracker.tracked_stracks)
            lost = len(tracker.lost_stracks)
            print(f"Frame {frame_id}: total={total_fps:.1f} fps, det={det_fps:.1f}, gmc={gmc_fps:.1f}, match={match_fps:.1f}, kalman={kalman_fps:.1f} | tracked={tracked}, lost={lost}")

        frame_id += 1

    cap.release()
    print(f"\nTotal frames: {frame_id}")


if __name__ == "__main__":
    args = make_parser().parse_args([])  # Parse empty to get defaults
    args.exp_file = 'yolox/exps/example/mot/yolox_s_mix_det.py'
    args.ckpt = 'pretrained/bytetrack_s_mot17.pth.tar'
    args.path = 'video/前保_30s.mp4'
    args.conf = 0.5
    args.fp16 = True
    args.fuse = True
    args.device = "gpu"
    args.ablation = False
    args.mot20 = not args.fuse_score

    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
