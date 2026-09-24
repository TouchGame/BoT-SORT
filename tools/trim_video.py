import cv2
import sys

def trim_video(input_path, output_path, seconds=30):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Cannot open {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input: {input_path}")
    print(f"  Resolution: {w}x{h}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")

    target_frames = int(fps * seconds)
    print(f"  Extracting {target_frames} frames ({seconds}s)...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    count = 0
    for i in range(target_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"  Reached end at frame {i}")
            break
        out.write(frame)
        count += 1

    cap.release()
    out.release()
    print(f"  Saved to {output_path} ({count} frames)")

if __name__ == "__main__":
    trim_video("video/前保装配9月8日/02010003375000000.mp4", "video/前保_30s.mp4", seconds=30)
