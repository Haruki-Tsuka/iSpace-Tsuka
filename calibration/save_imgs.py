import cv2
from utils.realsense_manager import RealSenseManager

rs_manager = RealSenseManager(640, 480, 30)

img_counter = 0

while True:
    rs_manager.update()
    frame = rs_manager.get_img()
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        img_name = f"images/frame_{img_counter}.png"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} written!")
        img_counter += 1