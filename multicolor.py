import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

elephant_url = "https://th.bing.com/th/id/OIP.LucytKBwbt8H0ATki-FbhQHaE7?w=282&h=188&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3"
elephant = load_image_from_url(elephant_url).convert("RGB")
elephant_np = np.array(elephant)

R, G, B = elephant_np[:, :, 0], elephant_np[:, :, 1], elephant_np[:, :, 2]

red_img = np.zeros_like(elephant_np)
green_img = np.zeros_like(elephant_np)
blue_img = np.zeros_like(elephant_np)

red_img[:, :, 0] = R
green_img[:, :, 1] = G
blue_img[:, :, 2] = B


plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(elephant_np)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(red_img)
plt.title("Red Channel Emphasis")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(green_img)
plt.title("Green Channel Emphasis")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(blue_img)
plt.title("Blue Channel Emphasis")
plt.axis("off")

plt.tight_layout()
plt.show()


elephant_gray = elephant.convert("L")
elephant_gray_np = np.array(elephant_gray)

plt.figure(figsize=(6, 5))
plt.imshow(elephant_gray_np, cmap="viridis")  
plt.title("Colormapped Grayscale")
plt.axis("off")
plt.colorbar()
plt.show()