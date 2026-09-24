import cv2, time

for path in ['前保_30s_h264.mp4', 'walker1.mp4']:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = 0
    t0 = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        n = n + 1
        if n >= 300:
            break
    dt = time.time() - t0
    print(path + ' 文件总帧数=' + str(total) + ' 读取' + str(n) + '帧 纯读取速度=' + format(n / dt, '.1f') + ' FPS')
    cap.release()