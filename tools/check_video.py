import cv2

for path in ["walker1.mp4", "前保_30s_h264.mp4"]:   # 改成你的两个文件名
    cap = cv2.VideoCapture(path)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)])
    print(path)
    print("  解码后端:", cap.getBackendName())
    print("  分辨率:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("  元数据帧率:", cap.get(cv2.CAP_PROP_FPS))
    print("  编码格式:", codec)
    cap.release()