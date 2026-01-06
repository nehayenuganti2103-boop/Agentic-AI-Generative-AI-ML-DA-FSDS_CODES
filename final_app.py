import streamlit as st 
import numpy as np 
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

st.set_page_config(page_title="Elephant Image Processor", layout="wide")

st.title("Elephant Image - Multi-Color Channel Visualizer")

@st.cache_data 
def load_image():
    url = r"https://th.bing.com/th/id/OIP.LucytKBwbt8H0ATki-FbhQHaE7?w=282&h=188&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3"
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

elephant = load_image()
st.image(elephant, caption="Original Elephant Image", use_container_width=True)

elephant_np = np.array(elephant)
R, G, B = elephant_np[:, :, 0], elephant_np[:, :, 1], elephant_np[:, :, 2]

red_img = np.zeros_like(elephant_np)
green_img = np.zeros_like(elephant_np)
blue_img = np.zeros_like(elephant_np)

red_img[:, :, 0] = R
green_img[:, :, 1] = G
blue_img[:, :, 2] = B

st.subheader("RGB Channel Visualization")
col1, col2, col3 = st.columns(3)

with col1:
    st.image(red_img, caption="Red Channel", use_container_width=True)

with col2:
    st.image(green_img, caption="Green Channel", use_container_width=True)

with col3:
    st.image(blue_img, caption="Blue Channel", use_container_width=True)

st.subheader("Colormapped Grayscale Image")

colormap = st.selectbox(
    "Choose a Matplotlib colormap",
    ["viridis", "plasma", "inferno", "magma", "cividis", "hot", "cool", "gray"]
)

elephant_gray = elephant.convert("L")
elephant_gray_np = np.array(elephant_gray)

fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(elephant_gray_np, cmap=colormap)
plt.axis("off")
st.pyplot(fig)
