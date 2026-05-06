import kachaka_api
from PIL import Image
import io

client = kachaka_api.KachakaApiClient('192.168.1.164:26400')

map = client.get_png_map()
print(map.name)
print(map.resolution, map.width, map.height)
print(map.origin)

img = Image.open(io.BytesIO(map.data))
img.save('map.png')