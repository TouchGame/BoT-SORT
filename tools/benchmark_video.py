import cv2
import time
import sys

def benchmark_video(path, name):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Cannot open {path}")
        return

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"{name}:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")

    start = time.time()
    count = 0
    for i in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
    elapsed = time.time() - start

    print(f"  Decoded {count} frames in {elapsed:.2f}s")
    print(f"  Decode FPS: {count/elapsed:.2f}")
    print()
    cap.release()

if __name__ == "__main__":
    print("Video Decode Benchmark")
    print("=" * 50)
    print()

    benchmark_video("video/前保装配9月8日/02010003375000000.mp4", "前保装配")
    benchmark_video("video/walker1.mp4", "walker1")
