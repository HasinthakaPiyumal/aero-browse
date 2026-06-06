import mmap
import os
import numpy as np
import cv2
import torch
from src.config import SHM_NAME, SHM_SIZE, FRAME_WIDTH, FRAME_HEIGHT

class FrameConsumer:
    def __init__(self):
        self.shm = mmap.mmap(-1, SHM_SIZE, tagname=SHM_NAME) if os.name == 'nt' else mmap.mmap(-1, SHM_SIZE)

    def get_tensor_from_shm(self):
        self.shm.seek(0)
        raw_bytes = self.shm.read(SHM_SIZE)
        
        frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 4))
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0
        return tensor