import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from typing import Optional

from tracker import matching
from tracker.gmc import GMC
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter import KalmanFilter
from tracker.employee_registry import EmployeeRegistry, EmployeeRegistryConfig
from tracker.face_registry import FaceRegistry, FaceRegistryConfig, FaceAnalysisWrapper

from fast_reid.fast_reid_interfece import FastReIDInterface

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger('bot_sort')

# Occlusion gating: boxes covered by nearer boxes beyond the configured ratio
# get no employee-gallery writes for that frame.
OCCLUSION_BOTTOM_MARGIN_RATIO = 0.03  # Bottom-edge difference below this is "same depth"
OCCLUSION_AREA_MARGIN_RATIO = 0.10    # Same-depth fallback: nearer box must be 10% larger


def _occlusion_ratio(box, other_boxes):
    """Fraction of `box` (xyxy) covered by boxes judged nearer.

    A box is nearer when its bottom edge projects lower in the image (ground-plane
    depth cue); same-depth overlaps only count for a clearly larger box. Multiple
    occluders sum up, capped at 1.0.
    """
    x1, y1, x2, y2 = box[:4]
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return 0.0
    area = w * h
    bottom_margin = OCCLUSION_BOTTOM_MARGIN_RATIO * h
    covered = 0.0
    for ob in other_boxes:
        ox1, oy1, ox2, oy2 = ob[:4]
        ix1, iy1 = max(x1, ox1), max(y1, oy1)
        ix2, iy2 = min(x2, ox2), min(y2, oy2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        if oy2 - y2 > bottom_margin:
            nearer = True
        elif oy2 - y2 >= -bottom_margin:
            nearer = (ox2 - ox1) * (oy2 - oy1) > area * (1.0 + OCCLUSION_AREA_MARGIN_RATIO)
        else:
            nearer = False
        if nearer:
            covered += (ix2 - ix1) * (iy2 - iy1)
    return min(covered / area, 1.0)


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        # EMA since the last (re)activation; stable identity re-checks query with this so a
        # re-found track is never matched on the appearance of a suspected earlier identity
        self.stable_smooth_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

        # Employee identity tracking
        self.employee_id = None
        self.birth_pos = None  # Foot point at track birth; anchors the position bonus at stable re-check
        self.stable_count = 0
        self.face_pending_verify = False  # True when identity was assigned body-only, awaiting face confirmation
        self.occluded = False  # True when this frame's matched detection is judged heavily occluded

        # Face feature buffer: accumulate face features before identity assignment
        self.face_features_buffer = []  # [(feature, frame_id, quality_score, pose_label, face_size), ...]

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

        if self.stable_smooth_feat is None:
            self.stable_smooth_feat = feat
        else:
            self.stable_smooth_feat = self.alpha * self.stable_smooth_feat + (1 - self.alpha) * feat
        self.stable_smooth_feat /= np.linalg.norm(self.stable_smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        # Restart the window here: everything before this re-detection belongs to a possibly
        # mis-assigned identity, so the stable re-check must only see post-reactivation frames
        self.stable_smooth_feat = None
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.occluded = new_track.occluded

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.stable_count += 1

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.occluded = new_track.occluded

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BoTSORT(object):
    def __init__(self, args, frame_rate=30, video_fps=None):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        self.args = args

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.frame_rate = frame_rate
        self.video_fps = video_fps or frame_rate  # For timestamp calculation
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        if args.with_reid:
            self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)

        self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])

        # Employee registry for long-term identity management
        self.with_employee_registry = getattr(args, 'with_employee_registry', False)
        if self.with_employee_registry:
            self.employee_registry = EmployeeRegistry()
            self.min_track_length = self.employee_registry.config.MIN_TRACK_LENGTH
            self.max_occlusion_ratio = self.employee_registry.config.MAX_OCCLUSION_RATIO
            self.employee_presence = {}  # {employee_id: last_frame_seen}
        else:
            self.employee_registry = None
            self.employee_presence = {}

        # Face registry for face-based identity verification
        self.with_face_registry = getattr(args, 'with_face_registry', False)
        if self.with_face_registry:
            self.face_registry = FaceRegistry()
            self.face_analysis = FaceAnalysisWrapper()
        else:
            self.face_registry = None
            self.face_analysis = None

        # For FRAME SUMMARY change detection
        self._last_summary_state = None

    @property
    def video_time(self):
        """Get current video time in seconds."""
        return self.frame_id / self.video_fps

    def update(self, output_results, img):
        self.frame_id += 1
        self._current_img = img  # Store for face detection
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, -1]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
                classes = output_results[:, -1]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]

        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        '''Extract embeddings '''
        if self.args.with_reid:
            features_keep = self.encoder.inference(img, dets)

        if len(dets) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                              (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
            if self.with_employee_registry:
                for det in detections:
                    det.occluded = bool(_occlusion_ratio(det.tlbr, bboxes) > self.max_occlusion_ratio)
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0
        else:
            dists = ious_dists

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                old_smooth_feat = track.smooth_feat.copy() if track.smooth_feat is not None else None
                was_lost_for = self.frame_id - track.frame_id
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
                track._pre_reactivation_smooth_feat = old_smooth_feat
                logger.debug(f"[{self.video_time:.2f}s] TRACK RE-ACTIVATED: track_id={track.track_id}, employee_id={track.employee_id}, "
                           f"was_lost_for={was_lost_for} frames")

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                old_smooth_feat = track.smooth_feat.copy() if track.smooth_feat is not None else None
                was_lost_for = self.frame_id - track.frame_id
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
                track._pre_reactivation_smooth_feat = old_smooth_feat
                logger.debug(f"[{self.video_time:.2f}s] TRACK RE-ACTIVATED (2nd assoc): track_id={track.track_id}, "
                           f"employee_id={track.employee_id}, was_lost_for={was_lost_for} frames")

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
                logger.debug(f"[{self.video_time:.2f}s] TRACK LOST: track_id={track.track_id}, employee_id={track.employee_id}, "
                           f"stable_count={track.stable_count}")

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            if self.with_employee_registry:
                track.birth_pos = self.employee_registry.foot_point(track.tlbr)
            logger.debug(f"[{self.video_time:.2f}s] NEW TRACK: track_id={track.track_id}, score={track.score:.3f}")

            # Try to identify new track via employee registry
            if self.with_employee_registry and track.curr_feat is not None:
                # Get set of employee_ids that already have active or lost tracks
                active_employee_ids = {t.employee_id for t in self.tracked_stracks + activated_starcks if t.employee_id is not None}
                query_pos = self.employee_registry.foot_point(track.tlbr)

                all_distances = []
                for emp_id_iter, gallery in self.employee_registry.galleries.items():
                    if emp_id_iter in active_employee_ids:
                        continue
                    if len(gallery) < self.employee_registry.config.MIN_GALLERY_SIZE:
                        continue
                    gallery_features = np.stack([g[0] for g in gallery])
                    gallery_frames = np.array([g[1] for g in gallery])
                    distances = self.employee_registry._cosine_distance(track.smooth_feat, gallery_features)
                    age = self.frame_id - gallery_frames
                    time_weights = np.maximum(np.exp(-self.employee_registry.config.TIME_WEIGHT_DECAY * age),
                                              self.employee_registry.config.MIN_TIME_WEIGHT)
                    weighted_distances = distances / time_weights
                    min_dist = np.min(weighted_distances)
                    bonus = self.employee_registry.position_bonus(emp_id_iter, query_pos, self.frame_id)
                    all_distances.append((emp_id_iter, min_dist - bonus, bonus))

                emp_id, dist, feat_idx = self.employee_registry.identify(track.smooth_feat, exclude_ids=active_employee_ids, current_frame=self.frame_id, query_pos=query_pos)
                all_distances.sort(key=lambda x: x[1])

                # Face detection for new track
                face_emp_id, face_dist = None, float('inf')
                face_all_distances = []
                face_pose_label = 'front'
                if self.with_face_registry and self.face_analysis.is_available():
                    person_bbox = track.tlbr
                    face_result = self.face_analysis.extract_face(self._current_img, person_bbox, self.video_time)
                    if face_result is not None:
                        face_feat, face_score, face_pose_label, face_yaw, face_size = face_result
                        # Add to buffer for later use if not matched immediately
                        track.face_features_buffer.append((face_feat, self.frame_id, face_score, face_pose_label, face_size))
                        logger.debug(f"[{self.video_time:.2f}s] FACE BUFFERED: track_id={track.track_id}, "
                                   f"pose={face_pose_label}, score={face_score:.2f}, "
                                   f"buffer_size={len(track.face_features_buffer)}, frame={self.frame_id}")
                        face_emp_id, face_dist, face_feat_idx, face_all_distances = self.face_registry.identify(
                            face_feat, exclude_ids=active_employee_ids, current_frame=self.frame_id, return_all=True,
                            face_size=face_size
                        )
                        # If face matches, prefer face identity
                        if face_emp_id is not None:
                            emp_id = face_emp_id
                            dist = face_dist
                            self.face_registry.refresh_feature(face_emp_id, face_feat_idx, face_feat, self.frame_id, face_pose_label)
                            self.face_registry.add_features(face_emp_id, face_feat, self.frame_id, face_pose_label, self.video_time)

                logger.debug(f"[{self.video_time:.2f}s] REGISTRY CHECK: track_id={track.track_id}, matched={emp_id}, "
                           f"distance={dist:.4f}, all_distances(emp,adj,bonus)={[(e, f'{d:.4f}', f'{b:.3f}') for e, d, b in all_distances]}, "
                           f"face_match={face_emp_id}, face_dist={face_dist:.4f}, "
                           f"face_all={[(e, f'{d:.4f}') for e, d in face_all_distances]}, "
                           f"registry_size={len(self.employee_registry.list_employees())}, active_employees={active_employee_ids}")
                if emp_id is not None:
                    track.employee_id = emp_id
                    # Set face_pending_verify based on whether face confirmed the match
                    if face_emp_id is not None and face_emp_id == emp_id:
                        track.face_pending_verify = False  # Face confirmed
                    else:
                        track.face_pending_verify = True  # Body-only, needs face verification
                    if not track.occluded:
                        self.employee_registry.refresh_feature(emp_id, feat_idx, track.curr_feat, self.frame_id)
                        pruned = self.employee_registry.prune_gallery(emp_id, track.curr_feat)
                    # Update face registry if face was detected
                    if face_emp_id is not None and len(track.face_features_buffer) > 0:
                        best_face_feat = track.face_features_buffer[0][0]
                        best_face_pose = track.face_features_buffer[0][3]
                        self.face_registry.add_features(emp_id, best_face_feat, self.frame_id, best_face_pose, self.video_time)
                    track.face_features_buffer = []

            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
                logger.debug(f"[{self.video_time:.2f}s] TRACK REMOVED: track_id={track.track_id}, employee_id={track.employee_id}, "
                           f"lost_frames={self.frame_id - track.end_frame}")

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        """ Step 5.4: Verify re-activated tracks (after all frame states are settled) """
        for track in refind_stracks:
            old_smooth_feat = getattr(track, '_pre_reactivation_smooth_feat', None)
            self._verify_reactivated_identity(track, old_smooth_feat)

        """ Step 5.5: Collect face features for all tracked tracks """
        if self.with_face_registry and self.face_analysis.is_available():
            self._collect_face_features()

        """ Step 5.6: Update employee presence history and last positions """
        if self.with_employee_registry:
            for t in self.tracked_stracks:
                if t.employee_id is not None:
                    self.employee_presence[t.employee_id] = self.frame_id
                    self.employee_registry.update_position(t.employee_id, self.employee_registry.foot_point(t.tlbr), self.frame_id)

        """ Step 6: Collect features from stable tracks to employee registry """
        if self.with_employee_registry:
            self._collect_features_to_registry()

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]

        # Frame summary - only print when state changes
        if self.with_employee_registry:
            emp_ids = [t.employee_id for t in output_stracks if t.employee_id]
            current_state = (len(output_stracks), len(self.lost_stracks), len(self.employee_registry.list_employees()), len(emp_ids))
            if current_state != self._last_summary_state:
                logger.debug(f"[{self.video_time:.2f}s] FRAME SUMMARY: tracked={current_state[0]}, "
                           f"lost={current_state[1]}, registry_employees={current_state[2]}, "
                           f"identified_tracks={current_state[3]}")
                self._last_summary_state = current_state


        return output_stracks

    def _collect_face_features(self):
        """
        Collect face features for tracked tracks.
        - Skip frames: only detect every 5 frames
        - For tracks without identity: store in buffer for later matching
        - For tracks with identity: add directly to gallery if not full
        """
        # Skip frames: only detect every 5 frames
        if self.frame_id % 5 != 0:
            return

        for track in self.tracked_stracks:
            # Check if we should collect face for this track
            if track.employee_id is not None:
                # Track has identity - check if all slots are full
                pose_galleries = self.face_registry.galleries.get(track.employee_id, {'front': [], 'left': [], 'right': []})
                total_features = sum(len(pose_galleries[p]) for p in ['front', 'left', 'right'])
                if total_features >= self.face_registry.config.MAX_GALLERY_SIZE:
                    continue  # Gallery full, skip detection

            person_bbox = track.tlbr
            face_result = self.face_analysis.extract_face(self._current_img, person_bbox, self.video_time)

            if face_result is None:
                continue

            face_feat, face_score, face_pose_label, face_yaw, face_size = face_result

            if track.employee_id is not None and not track.face_pending_verify:
                # Track has identity (already verified) - add directly to gallery
                self.face_registry.add_features(track.employee_id, face_feat, self.frame_id, face_pose_label, self.video_time)
            else:
                # Track without identity OR pending face verification - store in buffer for matching
                # Deduplication: skip if too similar to existing buffered features
                is_duplicate = False
                for existing_feat, _, _, _, _ in track.face_features_buffer:
                    similarity = np.dot(face_feat, existing_feat) / (np.linalg.norm(face_feat) * np.linalg.norm(existing_feat) + 1e-8)
                    if similarity > 0.85:
                        is_duplicate = True
                        break

                if is_duplicate:
                    continue

                # Add to buffer
                track.face_features_buffer.append((face_feat, self.frame_id, face_score, face_pose_label, face_size))
                logger.debug(f"[{self.video_time:.2f}s] FACE BUFFERED: track_id={track.track_id}, "
                           f"pose={face_pose_label}, score={face_score:.2f}, "
                           f"buffer_size={len(track.face_features_buffer)}, frame={self.frame_id}")

                # Limit buffer size (keep best quality faces)
                max_buffer_size = 5
                if len(track.face_features_buffer) > max_buffer_size:
                    # Sort by quality score and keep top N
                    track.face_features_buffer.sort(key=lambda x: x[2], reverse=True)
                    track.face_features_buffer = track.face_features_buffer[:max_buffer_size]

    def _collect_features_to_registry(self):
        """
        Collect features from stable tracks to the employee registry.

        For tracks that have been tracked for enough frames:
        - If track already has employee_id: add features to that employee's gallery
        - If track doesn't have employee_id: try to match against registry,
          or create a new employee if no match found
        - If face registry enabled: detect face and apply cascade verification
        """
        for track in self.tracked_stracks:
            if track.stable_count < self.min_track_length:
                continue
            if track.curr_feat is None:
                continue

            # Get active employee IDs (for exclusion)
            # Include: currently tracked employees + employees seen since this track was born
            # Exclude: this track's previous employee_id (if any) to allow re-matching
            active_employee_ids = {t.employee_id for t in self.tracked_stracks if t.employee_id is not None}
            for emp_id, last_frame in self.employee_presence.items():
                if last_frame >= track.start_frame:
                    active_employee_ids.add(emp_id)
            # Allow track to re-match its previous employee_id — only if no other
            # active track currently claims it
            if hasattr(track, '_previous_employee_id') and track._previous_employee_id is not None:
                claimed_by_other = any(
                    t is not track and t.employee_id == track._previous_employee_id
                    for t in self.tracked_stracks
                )
                if not claimed_by_other:
                    active_employee_ids.discard(track._previous_employee_id)

            # Body feature matching
            body_emp_id, body_dist, body_feat_idx = None, float('inf'), -1
            if track.employee_id is None:
                # Try to identify via body feature (use the birth-time position, not the moved box)
                # Query with the windowed feature (since last re-activation) so a re-found track
                # is not matched on the appearance of its earlier, possibly wrong identity
                query_feat = track.stable_smooth_feat if track.stable_smooth_feat is not None else track.smooth_feat
                query_pos = track.birth_pos if track.birth_pos is not None else self.employee_registry.foot_point(track.tlbr)
                all_distances = []
                for emp_id, gallery in self.employee_registry.galleries.items():
                    if emp_id in active_employee_ids:
                        continue
                    if len(gallery) < self.employee_registry.config.MIN_GALLERY_SIZE:
                        continue
                    gallery_features = np.stack([g[0] for g in gallery])
                    gallery_frames = np.array([g[1] for g in gallery])
                    distances = self.employee_registry._cosine_distance(query_feat, gallery_features)
                    age = self.frame_id - gallery_frames
                    time_weights = np.maximum(np.exp(-self.employee_registry.config.TIME_WEIGHT_DECAY * age),
                                              self.employee_registry.config.MIN_TIME_WEIGHT)
                    weighted_distances = distances / time_weights
                    min_dist = np.min(weighted_distances)
                    bonus = self.employee_registry.position_bonus(emp_id, query_pos, track.start_frame)
                    all_distances.append((emp_id, min_dist - bonus, bonus))

                body_emp_id, body_dist, body_feat_idx = self.employee_registry.identify(
                    query_feat, exclude_ids=active_employee_ids, current_frame=self.frame_id, query_pos=query_pos,
                    query_frame=track.start_frame
                )
                all_distances.sort(key=lambda x: x[1])
                body_all_distances = all_distances
            else:
                body_all_distances = []

            # Face feature matching using accumulated buffer
            face_emp_id, face_dist, face_feat_idx, face_score = None, float('inf'), -1, 0.0
            face_feat = None
            face_pose_label = 'front'
            face_all_distances = []
            if self.with_face_registry and self.face_analysis.is_available() and len(track.face_features_buffer) > 0:
                # Match all buffered face features, pick best match
                face_active_ids = active_employee_ids.copy()
                # Allow face to match track's own identity for pending verification
                if track.face_pending_verify and track.employee_id is not None:
                    face_active_ids.discard(track.employee_id)
                face_emp_id, face_dist, face_feat_idx, face_feat, face_all_distances = self.face_registry.identify_best_match(
                    track.face_features_buffer, exclude_ids=face_active_ids, current_frame=self.frame_id
                )
                if face_feat is not None:
                    # Get quality score and pose label of best face
                    for buf_feat, buf_fid, buf_score, buf_pose, _ in track.face_features_buffer:
                        if np.array_equal(buf_feat, face_feat):
                            face_score = buf_score
                            face_pose_label = buf_pose
                            break

            # Cascade verification: determine final identity
            final_emp_id = self._cascade_verification(
                track, body_emp_id, body_dist, body_feat_idx,
                face_emp_id, face_dist, face_feat_idx, face_feat, face_score
            )

            # Log cascade verification result only when there's data
            if body_emp_id is not None or face_emp_id is not None or len(track.face_features_buffer) > 0:
                logger.debug(f"[{self.video_time:.2f}s] CASCADE: track_id={track.track_id}, "
                           f"body={body_emp_id}(dist={body_dist:.4f}), "
                           f"face={face_emp_id}(dist={face_dist:.4f}, score={face_score:.2f}, buffer={len(track.face_features_buffer)}), "
                           f"final={final_emp_id}")

            # Update registries based on final identity
            if track.employee_id is None and final_emp_id is not None:
                track.employee_id = final_emp_id
                # Set face_pending_verify based on whether face confirmed the match
                if face_emp_id is not None and face_emp_id == final_emp_id:
                    track.face_pending_verify = False  # Face confirmed
                else:
                    track.face_pending_verify = True  # Body-only, needs face verification later
                logger.debug(f"[{self.video_time:.2f}s] STABLE MATCH: track_id={track.track_id}, "
                           f"employee={final_emp_id}, "
                           f"body_match={body_emp_id}, body_dist={body_dist:.4f}, "
                           f"body_all(emp,adj,bonus)={[(e, f'{d:.4f}', f'{b:.3f}') for e, d, b in body_all_distances]}, "
                           f"face_match={face_emp_id}, face_dist={face_dist:.4f}, "
                           f"face_all={[(e, f'{d:.4f}') for e, d in face_all_distances]}, "
                           f"face_score={face_score:.2f}, face_buffer={len(track.face_features_buffer)}, "
                           f"face_pending={track.face_pending_verify}, "
                           f"registry_size={len(self.employee_registry.list_employees())}, "
                           f"active_employees={active_employee_ids}")
                # Update body registry
                if body_emp_id == final_emp_id and body_feat_idx >= 0:
                    if not track.occluded:
                        self.employee_registry.refresh_feature(final_emp_id, body_feat_idx, track.curr_feat, self.frame_id)
                        self.employee_registry.add_features(final_emp_id, track.curr_feat, self.frame_id)
                        self.employee_registry.prune_gallery(final_emp_id, track.curr_feat)
                elif body_emp_id is None:
                    # Body didn't match, create new employee in body registry
                    self.employee_registry.add_employee(final_emp_id)
                    if not track.occluded:
                        self.employee_registry.add_features(final_emp_id, track.curr_feat, self.frame_id)

                # Update face registry
                if face_feat is not None and face_emp_id == final_emp_id and face_feat_idx >= 0:
                    self.face_registry.refresh_feature(final_emp_id, face_feat_idx, face_feat, self.frame_id, face_pose_label)
                    self.face_registry.add_features(final_emp_id, face_feat, self.frame_id, face_pose_label, self.video_time)
                    self.face_registry.prune_gallery(final_emp_id, face_feat)
                elif face_feat is not None and face_emp_id is None:
                    # Face didn't match, add to face registry under final_emp_id
                    self.face_registry.add_features(final_emp_id, face_feat, self.frame_id, face_pose_label, self.video_time)

            elif track.employee_id is not None:
                # Track already has identity, add features to existing galleries
                if not track.occluded:
                    self.employee_registry.add_features(track.employee_id, track.curr_feat, self.frame_id)
                if face_feat is not None and not track.face_pending_verify:
                    self.face_registry.add_features(track.employee_id, face_feat, self.frame_id, face_pose_label, self.video_time)

                # Face pending verification: check if face confirms or contradicts assigned identity
                if track.face_pending_verify:
                    if face_emp_id is not None and face_emp_id == track.employee_id:
                        # Face confirms the body-assigned identity
                        track.face_pending_verify = False
                        logger.debug(f"[{self.video_time:.2f}s] FACE VERIFIED: track_id={track.track_id}, "
                                   f"employee={track.employee_id}, face_dist={face_dist:.4f}")
                    elif face_emp_id is not None and face_emp_id != track.employee_id:
                        # Face contradicts body-assigned identity — trust face (more reliable)
                        old_emp_id = track.employee_id
                        active_employee_ids.discard(old_emp_id)
                        if face_emp_id not in active_employee_ids:
                            # Reassign to face-matched identity
                            track.employee_id = face_emp_id
                            track.face_pending_verify = False
                            logger.debug(f"[{self.video_time:.2f}s] FACE REASSIGN: track_id={track.track_id}, "
                                       f"old_employee={old_emp_id}, new_employee={face_emp_id}, "
                                       f"face_dist={face_dist:.4f}")
                            # If the old identity was minted for this track and no other track
                            # holds it, remove it — it was a false employee
                            if old_emp_id == getattr(track, '_created_employee_id', None):
                                held_by_others = any(
                                    t is not track and t.employee_id == old_emp_id
                                    for t in self.tracked_stracks + self.lost_stracks
                                )
                                if not held_by_others:
                                    self.employee_registry.remove_employee(old_emp_id)
                                    self.employee_presence.pop(old_emp_id, None)
                                    self.face_registry.galleries.pop(old_emp_id, None)
                                    self.face_registry.metadata.pop(old_emp_id, None)
                                    logger.debug(f"[{self.video_time:.2f}s] EMPLOYEE REMOVED: employee={old_emp_id}, "
                                               f"reason=face_reassign_to={face_emp_id}, track_id={track.track_id}")
                                track._created_employee_id = None
                            # Add face feature to new employee's gallery
                            self.face_registry.add_features(face_emp_id, face_feat, self.frame_id, face_pose_label, self.video_time)
                        else:
                            # Face-matched employee is already actively tracked elsewhere — keep old identity
                            track.face_pending_verify = False
                            logger.debug(f"[{self.video_time:.2f}s] FACE CONFLICT IGNORED: track_id={track.track_id}, "
                                       f"employee={old_emp_id}, face_match={face_emp_id} (active elsewhere)")
                    elif len(track.face_features_buffer) > 0:
                        # Face didn't match anyone — check if assigned employee's face gallery is empty
                        pose_galleries = self.face_registry.galleries.get(track.employee_id, {'front': [], 'left': [], 'right': []})
                        total_face_features = sum(len(pose_galleries[p]) for p in ['front', 'left', 'right'])
                        if total_face_features == 0:
                            # Gallery is empty, no reference to compare against — seed with buffer face
                            best_buf_feat = max(track.face_features_buffer, key=lambda x: x[2])
                            buf_feat, buf_fid, buf_score, buf_pose, _ = best_buf_feat
                            self.face_registry.add_features(track.employee_id, buf_feat, buf_fid, buf_pose, self.video_time)
                            track.face_pending_verify = False
                            track.face_features_buffer = []
                            logger.debug(f"[{self.video_time:.2f}s] FACE SEEDED: track_id={track.track_id}, "
                                       f"employee={track.employee_id}, pose={buf_pose}, "
                                       f"(gallery was empty, no reference to verify against)")

            elif final_emp_id is None:
                # No match found — check if we can create a new employee
                # Use track_id as proxy for total tracks created (track_ids are monotonically increasing)
                total_employees = len(self.employee_registry.list_employees())
                if total_employees >= track.track_id:
                    # Don't create new employee, likely false match
                    pass
                else:
                    new_emp_id = f"emp_{total_employees + 1:03d}"
                    self.employee_registry.add_employee(new_emp_id)
                    track.employee_id = new_emp_id
                    track._created_employee_id = new_emp_id  # For cleanup if a face later reassigns this track
                    if not track.occluded:
                        self.employee_registry.add_features(new_emp_id, track.curr_feat, self.frame_id)
                    if face_feat is not None:
                        self.face_registry.add_features(new_emp_id, face_feat, self.frame_id, face_pose_label, self.video_time)
                        track.face_pending_verify = False
                    else:
                        track.face_pending_verify = True  # No face yet, needs verification later
                    logger.debug(f"[{self.video_time:.2f}s] NEW EMPLOYEE: track_id={track.track_id}, "
                               f"employee={new_emp_id}, "
                               f"body_match={body_emp_id}, body_dist={body_dist:.4f}, "
                               f"body_all(emp,adj,bonus)={[(e, f'{d:.4f}', f'{b:.3f}') for e, d, b in body_all_distances]}, "
                               f"face_match={face_emp_id}, face_dist={face_dist:.4f}, "
                               f"face_all={[(e, f'{d:.4f}') for e, d in face_all_distances]}, "
                               f"face_score={face_score:.2f}, face_buffer={len(track.face_features_buffer)}, "
                               f"registry_size={total_employees + 1}, "
                               f"active_employees={active_employee_ids}")

            # Clear face buffer after identity assignment
            if track.employee_id is not None:
                track.face_features_buffer = []

    def _cascade_verification(
        self, track, body_emp_id, body_dist, body_feat_idx,
        face_emp_id, face_dist, face_feat_idx, face_feat, face_score
    ) -> Optional[str]:
        """
        Cascade verification: combine body and face matching results.

        Logic:
        - If body matches and face matches same → confirm identity
        - If body matches and face matches different → trust face
        - If body matches and face not available → use body
        - If body doesn't match and face matches → use face
        - If neither matches → return None (will create new employee)
        """
        body_match = body_emp_id is not None
        face_match = face_emp_id is not None and face_feat is not None

        if body_match and face_match:
            if body_emp_id == face_emp_id:
                # Both agree
                return body_emp_id
            else:
                # Conflict: trust face (more reliable)
                return face_emp_id
        elif body_match:
            return body_emp_id
        elif face_match:
            return face_emp_id
        else:
            return None

    def _remove_lost_tracks_holding(self, employee_id, keeper_track):
        """
        One employee, one track: remove lost tracks still holding this identity.
        Their detections will be treated as new tracks from now on.
        """
        losers = [t for t in self.lost_stracks
                  if t is not keeper_track and t.employee_id == employee_id]
        if not losers:
            return

        for t in losers:
            t.mark_removed()
            self.removed_stracks.append(t)
            logger.debug(f"[{self.video_time:.2f}s] TRACK REMOVED: track_id={t.track_id}, employee_id={t.employee_id}, "
                       f"lost_frames={self.frame_id - t.end_frame}, reason=identity_taken_by_track={keeper_track.track_id}")

        self.lost_stracks = [t for t in self.lost_stracks if t.state != TrackState.Removed]

    def _verify_reactivated_identity(self, track, old_smooth_feat):
        """
        Verify identity after track re-activation.
        Uses curr_feat (current frame) against old smooth_feat and employee gallery.
        If verification fails, clears identity and tries to re-match from lost employees.
        """
        if not self.with_employee_registry or track.employee_id is None:
            return

        old_emp_id = track.employee_id

        if track.curr_feat is None or old_smooth_feat is None:
            return

        verify_thresh = self.employee_registry.config.REID_VERIFY_THRESHOLD
        verify_thresh_b = self.employee_registry.config.REID_VERIFY_THRESHOLD_B

        # Check A: curr_feat vs old smooth_feat (before re_activate updated it)
        old_smooth_norm = old_smooth_feat / (np.linalg.norm(old_smooth_feat) + 1e-8)
        curr_norm = track.curr_feat / (np.linalg.norm(track.curr_feat) + 1e-8)
        feat_change_dist = 1.0 - np.dot(curr_norm, old_smooth_norm)

        # Check B: curr_feat vs employee gallery
        min_gallery_dist = float('inf')
        gallery = self.employee_registry.galleries.get(old_emp_id, [])
        if len(gallery) >= self.employee_registry.config.MIN_GALLERY_SIZE:
            gallery_features = np.stack([g[0] for g in gallery])
            distances = self.employee_registry._cosine_distance(track.curr_feat, gallery_features)
            min_gallery_dist = float(np.min(distances))

        check_a_fail = feat_change_dist > verify_thresh
        check_b_fail = min_gallery_dist > verify_thresh_b

        logger.debug(f"[{self.video_time:.2f}s] VERIFY CHECK: track_id={track.track_id}, employee={old_emp_id}, "
                   f"checkA_dist={feat_change_dist:.4f}({'FAIL' if check_a_fail else 'OK'}), "
                   f"checkB_dist={min_gallery_dist:.4f}({'FAIL' if check_b_fail else 'OK'}), "
                   f"thresh=({verify_thresh}, {verify_thresh_b})")

        if not check_a_fail and not check_b_fail:
            # Identity confirmed — remove any lost track still holding it, so
            # the same person's next detection becomes a new track instead of
            # a second holder of this employee
            self._remove_lost_tracks_holding(old_emp_id, track)

            # Body checks passed, but re-activation is high-risk — require face
            # confirmation on subsequent frames before trusting the identity again
            track.face_pending_verify = True
            logger.debug(f"[{self.video_time:.2f}s] FACE PENDING AFTER RE-ACTIVATION: track_id={track.track_id}, "
                       f"employee={old_emp_id}, body checks passed, awaiting face verification")
            return

        # Identity verification failed — try to re-match from lost employees
        old_emp_id_backup = track.employee_id
        track._previous_employee_id = old_emp_id_backup  # Save for re-matching
        track.employee_id = None
        active_employee_ids = {t.employee_id for t in self.tracked_stracks
                              if t.employee_id is not None}

        all_distances = []
        for emp_id_iter, emp_gallery in self.employee_registry.galleries.items():
            if emp_id_iter in active_employee_ids:
                continue
            if len(emp_gallery) < self.employee_registry.config.MIN_GALLERY_SIZE:
                continue
            g_features = np.stack([g[0] for g in emp_gallery])
            g_distances = self.employee_registry._cosine_distance(track.curr_feat, g_features)
            all_distances.append((emp_id_iter, float(np.min(g_distances))))
        all_distances.sort(key=lambda x: x[1])

        emp_id, dist, feat_idx = self.employee_registry.identify(
            track.curr_feat, exclude_ids=active_employee_ids, current_frame=self.frame_id
        )

        logger.debug(f"[{self.video_time:.2f}s] VERIFY REGISTRY CHECK: track_id={track.track_id}, "
                   f"old_emp={old_emp_id}, matched={emp_id}, distance={dist:.4f}, "
                   f"all_distances={[(e, f'{d:.4f}') for e, d in all_distances]}, "
                   f"registry_size={len(self.employee_registry.list_employees())}, "
                   f"active_employees={active_employee_ids}")

        if emp_id is not None:
            track.employee_id = emp_id
            track.face_pending_verify = True  # Body-only re-match, needs face verification
            self._remove_lost_tracks_holding(emp_id, track)
            if not track.occluded:
                self.employee_registry.refresh_feature(emp_id, feat_idx, track.curr_feat, self.frame_id)
                self.employee_registry.prune_gallery(emp_id, track.curr_feat)
        else:
            # No match found — mark as gray, wait for stable re-match
            # Don't create new employee immediately, let _collect_features_to_registry handle it
            # after a few frames with fresh features
            track.stable_count = 0
            track.face_features_buffer = []
            track.face_pending_verify = False
            logger.debug(f"[{self.video_time:.2f}s] VERIFY FAILED, MARKED GRAY: track_id={track.track_id}, "
                       f"old_employee={old_emp_id}, will retry after stabilization")


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


# # Alias for compatibility
# BoT_Sort = BoTSORT
