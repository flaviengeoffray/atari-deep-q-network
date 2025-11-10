import numpy as np
from PIL import Image


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess a single frame:
    - Convert to grayscale
    - Resize to 84x84
    - Normalize pixel values to [0, 1]
    """
    # Convert to grayscale
    gray_frame = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
    
    # Resize to 84x84
    img = Image.fromarray(gray_frame).resize((84, 84))
    resized_frame = np.array(img)
    
    # Normalize pixel values
    normalized_frame = resized_frame / 255.0
    
    return normalized_frame.astype(np.float32)
