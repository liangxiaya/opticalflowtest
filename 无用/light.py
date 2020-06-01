import cv2
from numpy import *

def draw_flow(im, flow, step=16):
    h, w = im.shape[:2]
    y, x = mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1)
    fx, fy = flow[int32(y), int32(x)].T

    # create line endpoints
    lines = vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = int32(lines)

    # create image and draw
    vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis,True
for x in range(1):
    video = cv2.VideoCapture("images/Rimg_%03d.png"%x)
    success, prvs = video.read()
    # print(video)
    prvs = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)

while success and cv2.waitKey(1) & 0xFF != ord('q'):
#等待1毫秒读取键键盘输入，最后一个字节是键盘的ASCII码。ord()返回字母的ASCII码
    success, frame = video.read()
    next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev=prvs, next=next, flow=None, pyr_scale=0.5, levels=10,
                                                winsize=50,
                                                iterations=1, poly_n=7, poly_sigma=1.2,
                                                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    frame,ret = draw_flow(next, flow)
    cv2.imshow('Optical flow', frame)