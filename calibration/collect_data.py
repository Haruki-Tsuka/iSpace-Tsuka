import cv2
import kachaka_api
from utils.realsense_manager import RealSenseManager
from utils.file_manager import CSVFileManager, ConfigManager

# manager = ConfigManager('setting.ini')
# ip = manager.get('KACHAKA', 'ip')

ip = '192.168.1.164'

if not ip:
    raise ValueError('KachakaIPが見つかりませんでした')

rs_manager = RealSenseManager(640, 480, 30)
csv_file = CSVFileManager(dir='csv', csv_name='kachaka_coords.csv', columns=['pix_x', 'pix_y', 'kac_x', 'kac_y', 'world_z'])
csv_file.create()

world_z = 0.12

count = 0

client = kachaka_api.KachakaApiClient(f'{ip}:26400')

def on_mouse_click(event, x, y, flags, param):
    global count
    if event == cv2.EVENT_LBUTTONDOWN:
        kachaka_pose = client.get_robot_pose()
        data = [x, y, kachaka_pose.x, kachaka_pose.y, world_z]
        csv_file.add(data)
        count += 1
        print(f'データ{count}: {data}')
        print(f'kachaka: {kachaka_pose}')

while True:
    rs_manager.update()
    im0 = rs_manager.get_img()
    cv2.namedWindow('Camera')
    cv2.setMouseCallback('Camera', on_mouse_click, param=im0)
    cv2.imshow('Camera', im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

rs_manager.close()
cv2.destroyAllWindows()
