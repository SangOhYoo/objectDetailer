import math
import numpy as np
import cv2
from PIL import Image, ImageEnhance

def calc_color_temperature(temp):
    """
    Converts a color temperature value (offset from 6500K) to RGB coefficients.
    Ref: sd-webui-bmab/processors/basic/final.py
    """
    white = (255.0, 254.11008387561782, 250.0419083427406)

    # BMAB uses a simpler scale logic: temp input is -2000 to 2000.
    # It adds this to 6500.
    temperature = (6500 + temp) / 100

    if temperature <= 66:
        red = 255.0
    else:
        red = float(temperature - 60)
        red = 329.698727446 * math.pow(red, -0.1332047592)
        if red < 0: red = 0
        if red > 255: red = 255

    if temperature <= 66:
        green = temperature
        green = 99.4708025861 * math.log(green) - 161.1195681661
    else:
        green = float(temperature - 60)
        green = 288.1221695283 * math.pow(green, -0.0755148492)
    if green < 0: green = 0
    if green > 255: green = 255

    if temperature >= 66:
        blue = 255.0
    else:
        if temperature <= 19:
            blue = 0.0
        else:
            blue = float(temperature - 10)
            blue = 138.5177312231 * math.log(blue) - 305.0447927307
            if blue < 0: blue = 0
            if blue > 255: blue = 255

    return red / white[0], green / white[1], blue / white[2]

def apply_bmab_basic(image_bgr, config):
    """
    Applies BMAB Basic effects: Contrast, Brightness, Sharpness, Color, Temperature, Noise.
    Input: BGR numpy array
    Output: BGR numpy array
    """
    # Convert to PIL RGB
    img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    
    # 1. Noise Alpha (Initial)
    noise_alpha = config.get('bmab_noise_alpha', 0.0)
    if noise_alpha > 0:
        # Generate noise
        w, h = img.size
        # BMAB Logic: Image.blend(image, img_noise, alpha)
        # Noise is random uniform 0-255? Ref check: util.generate_noise
        # "noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)"
        noise_np = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        img_noise = Image.fromarray(noise_np)
        img = Image.blend(img, img_noise, alpha=noise_alpha)

    # 2. Contrast
    contrast = config.get('bmab_contrast', 1.0)
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)

    # 3. Brightness
    brightness = config.get('bmab_brightness', 1.0)
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)

    # 4. Sharpness
    sharpness = config.get('bmab_sharpness', 1.0)
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpness)
        
    # 5. Color (Saturation)
    saturation = config.get('bmab_color_saturation', 1.0)
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation)

    # 6. Color Temperature
    temp = config.get('bmab_color_temperature', 0.0)
    if temp != 0.0:
        r_scale, g_scale, b_scale = calc_color_temperature(temp)
        # Apply per-pixel multiply
        # Use numpy for speed
        arr = np.array(img).astype(float)
        arr[:, :, 0] *= r_scale # R
        arr[:, :, 1] *= g_scale # G
        arr[:, :, 2] *= b_scale # B
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        
    # 7. Noise Alpha Final
    noise_alpha_final = config.get('bmab_noise_alpha_final', 0.0)
    if noise_alpha_final > 0:
        w, h = img.size
        noise_np = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        img_noise = Image.fromarray(noise_np)
        img = Image.blend(img, img_noise, alpha=noise_alpha_final)
        
    # Edge Enhancement (Optional - if set in config but might be handled separately or here)
    edge_strength = config.get('bmab_edge_strength', 0.0)
    if edge_strength > 0:
        low = config.get('bmab_edge_low', 50)
        high = config.get('bmab_edge_high', 200)
        
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low, high)
        
        # Darken logic from previous attempt or BMAB logic?
        # BMAB edge.py: 
        # "img = cv2.addWeighted(img, 1.0, edges_colored, -strength, 0)" -> Subtract edges?
        # Let's verify `sd_bmab/processors/basic/edge.py` if needed.
        # Check: view_file processors/basic/edge.py
        # For now, simplistic darken approach:
        
        arr = np.array(img).astype(float)
        mask = edges > 0
        arr[mask] -= (255.0 * edge_strength)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # Convert back to BGR
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
