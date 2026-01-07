import numpy as np
from PIL import Image as PILImage
from sensor_msgs.msg import Image
import base64
import io
import cv2

def opencv_to_ros_image(np_image):
    ros_image = Image()
    ros_image.height = np_image.shape[0]
    ros_image.width = np_image.shape[1]
    ros_image.encoding = "bgr8"
    ros_image.is_bigendian = 0
    ros_image.step = np_image.shape[1] * np_image.shape[2]  # width * channels
    ros_image.data = np_image.tobytes()
    return ros_image

def ros_image_to_opencv(ros_image):
    np_arr = np.frombuffer(ros_image.data, dtype=np.uint8)
    if ros_image.encoding == "rgb8" or ros_image.encoding == "bgr8":
        image = np_arr.reshape((ros_image.height, ros_image.width, 3))
    elif ros_image.encoding == "mono8":
        image = np_arr.reshape((ros_image.height, ros_image.width))
    else:
        raise ValueError(f"Unsupported encoding: {ros_image.encoding}")
    return image

def ros_image_to_pil(ros_image):
    # Convert raw image data to numpy array
    np_arr = np.frombuffer(ros_image.data, dtype=np.uint8)

    # Reshape based on encoding
    if ros_image.encoding == "rgb8" or ros_image.encoding == "bgr8":
        image = np_arr.reshape((ros_image.height, ros_image.width, 3))
        if ros_image.encoding == "bgr8":
            image = image[..., ::-1]  # Convert BGR to RGB for PIL
    elif ros_image.encoding == "mono8":
        image = np_arr.reshape((ros_image.height, ros_image.width))
    else:
        raise ValueError(f"Unsupported encoding: {ros_image.encoding}")

    return PILImage.fromarray(image)

def ros_image_to_base64(ros_image):
    # Step 1: Convert to PIL
    pil_img = ros_image_to_pil(ros_image)

    # Step 2: Save PIL image to BytesIO buffer as PNG
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")

    # Step 3: Base64 encode and decode to UTF-8 string
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded

def opencv_image_to_base64(cv_img) -> str:
    """
    Convert an OpenCV image (numpy array) to a base64-encoded PNG string.
    This is useful for visual-language models that accept images as base64 strings.
    """
    _, buffer = cv2.imencode('.png', cv_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

def get_vlm_img_message(img):
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}