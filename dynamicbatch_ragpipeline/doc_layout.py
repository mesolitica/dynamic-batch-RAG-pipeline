from doclayout_yolo import YOLOv10
from huggingface_hub import snapshot_download
from dynamicbatch_ragpipeline.env import args
from datetime import datetime
from io import BytesIO
from PIL import Image
import base64
import pymupdf
import numpy as np
import asyncio
import torch
import torchvision
import os
import logging
import time
import sys

id_to_names = {
    0: 'title', 
    1: 'plain text',
    2: 'abandon', 
    3: 'figure', 
    4: 'figure_caption', 
    5: 'table', 
    6: 'table_caption', 
    7: 'table_footnote', 
    8: 'isolate_formula', 
    9: 'formula_caption'
}

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

    model_dir = snapshot_download('juliozhao/DocLayout-YOLO-DocStructBench')
    model = YOLOv10(os.path.join(model_dir, 'doclayout_yolo_docstructbench_imgsz1024.pt'))
    
    if args.torch_compile:
        logging.info('enabling torch compile for doc layout')
        model.compile()
    
async def step():
    need_sleep = True
    while True:
        if need_sleep:
            await asyncio.sleep(args.dynamic_batching_microsleep)
    
        try:
            need_sleep = True
            batch = []
            while not step_queue.empty():
                try:
                    request = await asyncio.wait_for(step_queue.get(), timeout=1e-9)
                    batch.append(request)
                    if len(batch) >= args.dynamic_batching_doc_layout_batch_size:
                        need_sleep = False
                        break
                        
                except asyncio.TimeoutError:
                    break

            if not len(batch):
                continue

            futures = [batch[i][0] for i in range(len(batch))]
            input_img = [batch[i][1] for i in range(len(batch))]

            logging.debug(f'{str(datetime.now())} document layout step batch size of {len(input_img)}')

            with torch.no_grad():
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
            logging.error(e)
            try:
                futures = [batch[i][0] for i in range(len(batch))]
                for i in range(len(futures)):
                    if not futures[i].done():
                        futures[i].set_exception(e)
            except:
                pass

async def predict(
    file, 
    iou_threshold = 0.45, 
    ratio_x = 2.0, 
    ratio_y = 2.0, 
    request = None,
):
    doc = pymupdf.open(file)
    mat = pymupdf.Matrix(ratio_x, ratio_y)
    futures, images = [], []

    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        image = np.frombuffer(pix.samples_mv, dtype=np.uint8).reshape((pix.height, pix.width, 3)).copy()
        images.append(image)

        future = asyncio.Future()
        await step_queue.put((future, image))
        futures.append(future)
    
    before = time.time()
    results = await asyncio.gather(*futures)

    actual_results = []
    
    for i in range(len(results)):
        boxes, classes, scores = results[i]
        indices = torchvision.ops.nms(
            boxes=torch.Tensor(boxes), 
            scores=torch.Tensor(scores),
            iou_threshold=iou_threshold,
        )
        boxes, scores, classes = boxes[indices], scores[indices], classes[indices]
        if len(boxes.shape) == 1:
            boxes = np.expand_dims(boxes, 0)
            scores = np.expand_dims(scores, 0)
            classes = np.expand_dims(classes, 0)

        image = Image.fromarray(images[i])
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img = base64.b64encode(buffered.getvalue()).decode("utf-8")

        coordinates = boxes.int().cpu().numpy().tolist()
        classes = [id_to_names[int(c)] for c in classes]
        boxes = []
        for c in coordinates:
            x_min, y_min, x_max, y_max = c
            boxes.append({
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
            })
        sorted_indices = sorted(range(len(boxes)), key=lambda i: (boxes[i]['y_min'], boxes[i]['x_min']))
        sorted_boxes = [boxes[i] for i in sorted_indices]
        sorted_classes = [classes[i] for i in sorted_indices]
        d = {
            'classes': sorted_classes,
            'coordinates': sorted_boxes,
            'img': img,
        }
        actual_results.append(d)

    after = time.time()
    
    stats = {
        'total_page': len(images),
        'page_per_second': len(images) / (after - before),
    }
    return {
        'result': actual_results,
        'stats': stats,
    }

