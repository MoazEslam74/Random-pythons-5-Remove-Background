from rembg import remove
from PIL import Image

image_path='image.png'

input=Image.open(image_path)

output=remove(input)
output.save('RB_image.png')