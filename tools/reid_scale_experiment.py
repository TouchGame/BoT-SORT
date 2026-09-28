"""A/B comparison of ReID weights on the same crops.

Identity grouping is appearance-independent: BoTSORT(with_reid=False, strict IoU
matching, short lost-memory) produces track fragments, each split at sampling
gaps into "segments" (continuous presence runs). Every model is measured on the
same segments. Per box-height bucket it reports:
  - same-person distance: nearest crop of the same segment >= MIN_GAP apart
  - impostor distance: nearest crop of another segment in the same bucket
  - match rate: share of crops whose same-person distance < MATCH_THRESHOLD
A montage image is saved per segment for manual identity verification; flagged
segments can be excluded via an exclude file, or same-person segments merged
via an identities file ({"name": [sid, ...]}, reserved key "_junk" drops them).
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch

sys.path.append('.')

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess
from fast_reid.fast_reid_interfece import FastReIDInterface

MEANS = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MATCH_THRESHOLD = 0.4
MIN_GAP_SECONDS = 1.0
MAX_SPEED_PX = 30.0
MIN_SEGMENT = 5


class CompatExtractor(FastReIDInterface):
    """Also accepts old-style checkpoints that store the neck as heads.bnneck.*."""

    def __init__(self, config_file, weights_path, device, batch_size=8):
        super().__init__(config_file, weights_path, device, batch_size)
        ckpt = torch.load(weights_path, map_location='cpu')
        state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
        had_old_neck = any(k.startswith('heads.bnneck.') for k in state)
        renamed = {k.replace('heads.bnneck.', 'heads.bottleneck.0.'): v for k, v in state.items()}
        target = self.model.state_dict()
        keep = {k: v for k, v in renamed.items() if k in target and target[k].shape == v.shape}
        missing, unexpected = self.model.load_state_dict(keep, strict=False)
        benign = {'heads.weight', 'heads.classifier.weight'}
        self.incompatible = [k for k in missing if k not in benign]
        if had_old_neck and not self.incompatible:
            print('  [INFO] remapped heads.bnneck -> heads.bottleneck.0', flush=True)
        if self.incompatible:
            print(f'  [INCOMPATIBLE] missing keys after remap: {self.incompatible[:6]}', flush=True)


def build_tracker(device, fps):
    from tracker.bot_sort import BoTSORT

    args = SimpleNamespace(
        track_high_thresh=0.6, track_low_thresh=0.1, new_track_thresh=0.7,
        track_buffer=15, match_thresh=0.5, proximity_thresh=0.5,
        appearance_thresh=0.25, with_reid=False, fast_reid_config='',
        fast_reid_weights='', device=device, cmc_method='none',
        name='reid_exp', ablation=False, mot20=True,
        with_employee_registry=False, with_face_registry=False,
    )
    return BoTSORT(args, frame_rate=fps, video_fps=fps)


def run_tracking(video, exp_file, ckpt, device, max_frames, stride, min_h, fuse):
    exp = get_exp(exp_file, None)
    model = exp.get_model().to(device)
    state = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(state['model'])
    model.eval()
    if fuse:
        model = fuse_model(model)
    model = model.half()

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)
    tracker = build_tracker(device, fps)

    raw = {}
    frame_idx = 0
    while frame_idx < total:
        ok, frame = cap.read()
        if not ok:
            break
        img, _ = preproc(frame, exp.test_size, MEANS, STD)
        img = torch.from_numpy(img).unsqueeze(0).float().to(device).half()
        with torch.no_grad():
            out = model(img)
            out = postprocess(out, exp.num_classes, exp.test_conf, exp.nmsthre)
        scale = min(exp.test_size[0] / frame.shape[0], exp.test_size[1] / frame.shape[1])
        if out[0] is not None:
            dets = out[0].cpu().numpy()[:, :7]
            dets[:, :4] /= scale
        else:
            dets = np.zeros((0, 7))
        online = tracker.update(dets, frame)
        if frame_idx % stride == 0:
            for t in online:
                tlbr = t.tlbr
                h = float(tlbr[3] - tlbr[1])
                if h < min_h:
                    continue
                raw.setdefault(frame_idx, []).append((int(t.track_id), [float(v) for v in tlbr], h))
        frame_idx += 1
        if frame_idx % 400 == 0:
            print(f'  tracked {frame_idx}/{total}', flush=True)
    cap.release()
    return raw, fps


def build_segments(raw, stride, min_len):
    per_track = {}
    for f in sorted(raw):
        for tid, tlbr, h in raw[f]:
            per_track.setdefault(tid, []).append((f, tlbr, h))

    segments = {}
    sid = 0
    for tid in sorted(per_track):
        cur = []
        for item in per_track[tid]:
            if cur and item[0] - cur[-1][0] > stride:
                if len(cur) >= min_len:
                    segments[sid] = {'track': tid, 'items': cur}
                    sid += 1
                cur = []
            cur.append(item)
        if len(cur) >= min_len:
            segments[sid] = {'track': tid, 'items': cur}
            sid += 1
    return segments


def save_montages(video, segments, crop_dir):
    os.makedirs(crop_dir, exist_ok=True)
    by_frame = {}
    for sid, seg in segments.items():
        for f, tlbr, h in seg['items']:
            by_frame.setdefault(f, []).append(sid)
    cap = cv2.VideoCapture(video)
    frame_idx = 0
    crops = {}
    for target in sorted(by_frame):
        while frame_idx < target:
            if not cap.grab():
                break
            frame_idx += 1
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        items = {f: tlbr for f, tlbr, _ in segments[by_frame[target][0]]['items']}
        for sid in by_frame[target]:
            tlbr = dict((f, tlbr) for f, tlbr, _ in segments[sid]['items'])[target]
            x1, y1, x2, y2 = [int(round(v)) for v in tlbr]
            x1, y1 = max(0, x1), max(0, y1)
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                scale = 128.0 / crop.shape[0]
                crops.setdefault(sid, []).append(
                    (target, cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), 128))))
    cap.release()
    for sid, entries in crops.items():
        n = len(entries)
        pick = np.unique(np.linspace(0, n - 1, min(10, n)).astype(int))
        imgs = [entries[i] for i in pick]
        w = sum(im.shape[1] + 4 for _, im in imgs)
        sheet = np.full((132, w, 3), 30, dtype=np.uint8)
        x = 0
        for f, im in imgs:
            sheet[2:130, x:x + im.shape[1]] = im
            cv2.putText(sheet, str(f), (x + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            x += im.shape[1] + 4
        cv2.imwrite(os.path.join(crop_dir, f'seg_{sid:03d}_t{segments[sid]["track"]:04d}.jpg'), sheet)


def extract_features(video, segments, cfg, weights, device):
    extractor = CompatExtractor(cfg, weights, device)
    expected_dim = None
    by_frame = {}
    for sid, seg in segments.items():
        for f, tlbr, h in seg['items']:
            by_frame.setdefault(f, []).append((sid, tlbr))
    feats = {}
    cap = cv2.VideoCapture(video)
    frame_idx = 0
    for target in sorted(by_frame):
        while frame_idx < target:
            if not cap.grab():
                break
            frame_idx += 1
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        entries = by_frame[target]
        boxes = np.array([e[1] for e in entries], dtype=np.float64)
        vecs = extractor.inference(frame, boxes)
        for (sid, _), vec in zip(entries, vecs):
            feats[(sid, target)] = vec
    cap.release()
    incompatible = extractor.incompatible
    del extractor
    torch.cuda.empty_cache()
    return feats, incompatible


def load_feats(cache):
    data = np.load(cache, allow_pickle=False)
    keys, mat = data['keys'], data['feats']
    return {tuple(int(x) for x in str(k).split('_')): mat[i] for i, k in enumerate(keys)}


def save_feats(cache, feats):
    keys = np.array([f'{sid}_{f}' for sid, f in feats])
    mat = np.stack([feats[k] for k in feats])
    np.savez(cache, keys=keys, feats=mat)


def bucket_of(h, near_min, far_max):
    if h >= near_min:
        return 'near'
    if h <= far_max:
        return 'far'
    return 'mid'


def pair_bucket(ha, hb, near_min, far_max):
    rank = {'far': 0, 'mid': 1, 'near': 2}
    ba, bb = bucket_of(ha, near_min, far_max), bucket_of(hb, near_min, far_max)
    if rank[ba] > rank[bb]:
        ba, bb = bb, ba
    return f'{ba}-{bb}'


def analyze(feats, segments, near_min, far_max, min_gap, exclude, identities=None, max_ratio=8.0):
    groups = {}
    if identities:
        for name, sids in identities.items():
            if name == '_junk':
                continue
            items = []
            for sid in sids:
                if sid in exclude or sid not in segments:
                    continue
                items += [(sid, f, tlbr, h) for f, tlbr, h in segments[sid]['items'] if (sid, f) in feats]
            if items:
                groups[str(name)] = sorted(items, key=lambda it: it[1])
        listed = {sid for sids in identities.values() for sid in sids}
        for sid, seg in segments.items():
            if sid in exclude or sid in listed:
                continue
            items = [(sid, f, tlbr, h) for f, tlbr, h in sorted(seg['items']) if (sid, f) in feats]
            if items:
                groups[sid] = items
    else:
        for sid, seg in segments.items():
            if sid in exclude:
                continue
            items = [(sid, f, tlbr, h) for f, tlbr, h in sorted(seg['items']) if (sid, f) in feats]
            if items:
                groups[sid] = items

    item_ident = {}
    height_by_key = {}
    for name, items in groups.items():
        for sid, f, _, h in items:
            item_ident[(sid, f)] = name
            height_by_key[(sid, f)] = h

    same = {}
    own_min = {}
    for name, items in groups.items():
        for i in range(len(items)):
            sid_i, fi, box_i, hi = items[i]
            vi = feats[(sid_i, fi)]
            ci = ((box_i[0] + box_i[2]) / 2.0, (box_i[1] + box_i[3]) / 2.0)
            for j in range(i + 1, len(items)):
                sid_j, fj, box_j, hj = items[j]
                gap = fj - fi
                if gap < min_gap:
                    continue
                if max(hi, hj) > max_ratio * min(hi, hj):
                    continue
                if sid_i == sid_j:
                    cj = ((box_j[0] + box_j[2]) / 2.0, (box_j[1] + box_j[3]) / 2.0)
                    if np.hypot(ci[0] - cj[0], ci[1] - cj[1]) > MAX_SPEED_PX * gap + 150:
                        continue
                d = float(1.0 - np.dot(vi, feats[(sid_j, fj)]))
                same.setdefault(pair_bucket(hi, hj, near_min, far_max), []).append(d)
                for key in ((sid_i, fi), (sid_j, fj)):
                    if key not in own_min or d < own_min[key]:
                        own_min[key] = d

    keys = list(item_ident)
    feats_mat = np.stack([feats[k] for k in keys])
    dist = 1.0 - feats_mat @ feats_mat.T
    impostor = {}
    for idx, key in enumerate(keys):
        bucket = bucket_of(height_by_key[key], near_min, far_max)
        ident = item_ident[key]
        mask = np.array([item_ident[k] != ident and bucket_of(height_by_key[k], near_min, far_max) == bucket
                         for k in keys])
        if mask.any():
            impostor[key] = float(dist[idx][mask].min())

    p_report = {}
    for b in ('far-far', 'far-mid', 'far-near', 'mid-mid', 'mid-near', 'near-near'):
        ds = same.get(b, [])
        p_report[b] = {
            'n_pairs': len(ds),
            'med': round(float(np.median(ds)), 4) if ds else None,
            'p90': round(float(np.percentile(ds, 90)), 4) if ds else None,
        }

    q_report = {}
    for b in ('far', 'mid', 'near'):
        qkeys = [k for k in keys if bucket_of(height_by_key[k], near_min, far_max) == b]
        own = [own_min[k] for k in qkeys if k in own_min]
        imp = [impostor[k] for k in qkeys if k in impostor]
        q_report[b] = {
            'n_crops': len(qkeys),
            'n_scored': len(own),
            'match_rate_0.4': round(float(np.mean([v < MATCH_THRESHOLD for v in own])), 3) if own else None,
            'own_med': round(float(np.median(own)), 4) if own else None,
            'own_p90': round(float(np.percentile(own, 90)), 4) if own else None,
            'imp_med': round(float(np.median(imp)), 4) if imp else None,
            'imp_p05': round(float(np.percentile(imp, 5)), 4) if imp else None,
        }
    return {'pairs': p_report, 'queries': q_report}, own_min


def print_report(name, report):
    fmt = lambda v: '-' if v is None else f'{v:.3f}'
    print('  same-person pair distances by scale combination:', flush=True)
    print(f'  {"bucket":10s} {"pairs":>6s} {"med":>7s} {"p90":>7s}', flush=True)
    for b, r in report['pairs'].items():
        print(f'  {b:10s} {r["n_pairs"]:>6d} {fmt(r["med"]):>7s} {fmt(r["p90"]):>7s}', flush=True)
    print('  query-side (own = min distance to own identity, >=1s apart; imp = min to other identities, same bucket):',
          flush=True)
    print(f'  {"bucket":10s} {"crops":>6s} {"scored":>7s} {"rate<0.4":>9s} {"own_med":>8s} {"own_p90":>8s} '
          f'{"imp_med":>8s} {"imp_p05":>8s}', flush=True)
    for b, r in report['queries'].items():
        print(f'  {b:10s} {r["n_crops"]:>6d} {r["n_scored"]:>7d} {fmt(r["match_rate_0.4"]):>9s} '
              f'{fmt(r["own_med"]):>8s} {fmt(r["own_p90"]):>8s} {fmt(r["imp_med"]):>8s} {fmt(r["imp_p05"]):>8s}',
              flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--exp', default='yolox/exps/example/mot/yolox_s_mix_det.py')
    parser.add_argument('--ckpt', default='pretrained/bytetrack_s_mot17.pth.tar')
    parser.add_argument('--fuse', action='store_true')
    parser.add_argument('--model', action='append', required=True, help='name:config:weights')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--stride', type=int, default=15)
    parser.add_argument('--min-height', type=float, default=40.0)
    parser.add_argument('--near-min', type=float, default=450.0)
    parser.add_argument('--far-max', type=float, default=250.0)
    parser.add_argument('--exclude', default='')
    parser.add_argument('--identities', default='', help='JSON {identity: [segment ids]}, reserved key _junk drops segments')
    parser.add_argument('--max-ratio', type=float, default=8.0)
    parser.add_argument('--refresh-segments', action='store_true')
    parser.add_argument('--refresh-feats', action='store_true')
    parser.add_argument('--out', default='YOLOX_outputs/_reid_exp')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    seg_path = os.path.join(args.out, 'segments.json')
    if os.path.exists(seg_path) and not args.refresh_segments:
        with open(seg_path) as fh:
            blob = json.load(fh)
        segments = {int(k): v for k, v in blob['segments'].items()}
        fps = blob['fps']
        print(f'loaded {len(segments)} cached segments', flush=True)
    else:
        print('stage 1: detection + appearance-independent tracking (strict)', flush=True)
        raw, fps = run_tracking(args.video, args.exp, args.ckpt, device,
                                args.max_frames, args.stride, args.min_height, args.fuse)
        segments = build_segments(raw, args.stride, MIN_SEGMENT)
        with open(seg_path, 'w') as fh:
            json.dump({'segments': {str(k): v for k, v in segments.items()}, 'fps': fps}, fh)
        print(f'built {len(segments)} segments', flush=True)

    print(f'{"seg":>4s} {"track":>6s} {"n":>4s} {"frames":>13s} {"h_min":>6s} {"h_med":>6s} {"h_max":>6s}', flush=True)
    for sid in sorted(segments):
        hs = [h for _, _, h in segments[sid]['items']]
        fr = [f for f, _, _ in segments[sid]['items']]
        print(f'{sid:>4d} {segments[sid]["track"]:>6d} {len(hs):>4d} '
              f'{fr[0]:>6d}-{fr[-1]:<6d} {min(hs):>6.0f} {np.median(hs):>6.0f} {max(hs):>6.0f}', flush=True)

    save_montages(args.video, segments, os.path.join(args.out, 'montage'))

    exclude = set()
    if args.exclude and os.path.exists(args.exclude):
        with open(args.exclude) as fh:
            exclude = set(json.load(fh))
        print(f'excluding segments: {sorted(exclude)}', flush=True)

    identities = None
    if args.identities and os.path.exists(args.identities):
        with open(args.identities) as fh:
            identities = json.load(fh)
        print(f'using identities file: {args.identities} ({len(identities)} groups)', flush=True)

    min_gap = int(round(MIN_GAP_SECONDS * fps))
    results = {}
    for spec in args.model:
        name, cfg, weights = spec.split(':')
        print(f'\nmodel {name}: {cfg} + {weights}', flush=True)
        cache = os.path.join(args.out, f'feats_{name}.npz')
        if os.path.exists(cache) and not args.refresh_feats:
            feats = load_feats(cache)
            incompatible = []
        else:
            feats, incompatible = extract_features(args.video, segments, cfg, weights, device)
            save_feats(cache, feats)
        if incompatible:
            print(f'  SKIP {name}: checkpoint incompatible ({len(incompatible)} missing keys)', flush=True)
            results[name] = {'incompatible': True}
            continue
        report, _ = analyze(feats, segments, args.near_min, args.far_max, min_gap, exclude,
                            identities, args.max_ratio)
        results[name] = report
        print_report(name, report)

    with open(os.path.join(args.out, 'results.json'), 'w') as fh:
        json.dump({'exclude': sorted(exclude), 'results': results}, fh, indent=2, ensure_ascii=False)
    print(f'\nsaved -> {os.path.join(args.out, "results.json")}', flush=True)


if __name__ == '__main__':
    main()
