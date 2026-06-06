import asyncio
import time
import base64
import numpy as np
import cv2
import mmap
import os
from src.config import CHROME_PORT, SHM_NAME, SHM_SIZE, FRAME_WIDTH, FRAME_HEIGHT
from src.cdp_client import CDPClient
from src.shm_pipeline import FrameConsumer

class FrameProducer:
    def __init__(self):
        if os.name == 'nt':
            self.shm = mmap.mmap(-1, SHM_SIZE, tagname=SHM_NAME)
        else:
            self.shm = mmap.mmap(-1, SHM_SIZE)

    def write_to_shm(self, raw_rgba_bytes):
        self.shm.seek(0)
        self.shm.write(raw_rgba_bytes)

async def producer_loop(cdp_client: CDPClient, producer: FrameProducer):
    print("[Producer] Starting Frame Capture Loop...")
    
    await cdp_client.enable_headless_renderer()
    
    while True:
        start_time = time.perf_counter()
        
        response = await cdp_client.send_command("HeadlessExperimental.beginFrame", {
            "screenshot": {
                "format": "png"
            }
        })
        
        if "result" in response and "screenshotData" in response["result"]:
            img_b64 = response["result"]["screenshotData"]
            img_bytes = base64.b64decode(img_b64)
            
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            
            if img is not None:
                resized = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
                if resized.shape[2] == 3:
                    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
                
                producer.write_to_shm(resized.tobytes())
                
                end_time = time.perf_counter()
                print(f"[Producer] Frame written to SHM in {(end_time - start_time)*1000:.2f}ms")
        
        await asyncio.sleep(0.033)

async def consumer_loop(consumer: FrameConsumer):
    print("[Consumer] Starting VLA Inference Read Loop...")
    await asyncio.sleep(1)
    
    while True:
        start_time = time.perf_counter()
        
        tensor = consumer.get_tensor_from_shm()
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        print(f"[Consumer] Tensor Ready! Shape: {tensor.shape} | Extraction Latency: {latency_ms:.2f}ms")
        
        await asyncio.sleep(0.05)

async def main():
    cdp_client = CDPClient()
    producer = FrameProducer()
    consumer = FrameConsumer()
    
    try:
        await cdp_client.connect()
    except Exception as e:
        print(f"❌ Error: Chrome එක run වෙන්නේ නැහැ වගේ machan. පොඩ්ඩක් check කරන්න port 9222 open ද කියලා. Details: {e}")
        return

    try:
        await asyncio.gather(
            producer_loop(cdp_client, producer),
            consumer_loop(consumer)
        )
    except KeyboardInterrupt:
        print("\nStopping VeloVLA loops cleanly...")

if __name__ == "__main__":
    asyncio.run(main())