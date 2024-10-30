from doclayout_yolo import YOLOv10
from huggingface_hub import snapshot_download
from dynamicbatch_ragpipeline.env import args
from datetime import datetime
import pymupdf
import numpy as np
import asyncio
import torch
import os
import logging
import time

zoom_x = 3.0
zoom_y = 3.0
mat = pymupdf.Matrix(zoom_x, zoom_y)

model = None
device = 'cpu'
if args.accelerator_type == 'cuda':
    if not torch.cuda.is_available():
        logging.warning('CUDA is not available, fallback to CPU.')
    else:
        device = 'cuda'

step_queue = asyncio.Queue()

def load_model():
    global model

    if args.model_doc_layout == 'yolo10':
        model_dir = snapshot_download('juliozhao/DocLayout-YOLO-DocStructBench')
        model = YOLOv10(os.path.join(model_dir, 'doclayout_yolo_docstructbench_imgsz1024.pt'))
    
async def step():
    while True:
        await asyncio.sleep(args.dynamic_batching_microsleep)
    
        try:
            batch = []
            while not step_queue.empty():
                try:
                    request = await asyncio.wait_for(step_queue.get(), timeout=1e-4)
                    batch.append(request)
                    if len(batch) >= args.dynamic_batching_batch_size:
                        break
                except asyncio.TimeoutError:
                    break

            if not len(batch):
                continue

            futures = [batch[i][0] for i in range(len(batch))]
            input_img = [batch[i][1] for i in range(len(batch))]

            logging.info(f'{str(datetime.now())} step batch size of {len(input_img)}')

            det_res = model.predict(
                input_img,
                imgsz=1024,
                conf=0.25,
                device=device,
                batch=len(input_img)
            )

            for i in range(len(futures)):
                boxes = det_res[i].__dict__['boxes'].xyxy
                classes = det_res[i].__dict__['boxes'].cls
                scores = det_res[i].__dict__['boxes'].conf
                futures[i].set_result((boxes, classes, scores))

        except Exception as e:
            print(f"Error in step: {e}")
            futures = [batch[i][0] for i in range(len(batch))]
            for i in range(len(futures)):
                if not futures[i].done():
                    futures[i].set_exception(e)

async def predict(file, request):
    request.scope['request']['before_time_taken'] = time.time()
    doc = pymupdf.open(file)
    futures = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        image = np.frombuffer(pix.samples_mv, dtype=np.uint8).reshape((pix.height, pix.width, 3)).copy()
        future = asyncio.Future()
        await step_queue.put((future, image))
        futures.append(future)
    
    results = await asyncio.gather(*futures)

    request.scope['request']['total_page'] = len(futures)
    request.scope['request']['after_time_taken'] = time.time()

