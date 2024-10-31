from fastapi import FastAPI, Request
from fastapi import File, Form, UploadFile
from sse_starlette import EventSourceResponse
from dynamicbatch_ragpipeline.env import args
from dynamicbatch_ragpipeline.doc_layout import (
    load_model as doc_layout_load_model, 
    predict as doc_layout_predict,
    step as doc_layout_step,
)
from dynamicbatch_ragpipeline.ocr import (
    load_model as ocr_load_model, 
    predict as ocr_predict,
    prefill as ocr_prefill,
    step as ocr_step,
)
from dynamicbatch_ragpipeline.function import cleanup_cache
import asyncio
import logging
import uvicorn
import uuid
import time
import torch
import os
import tempfile
from pathlib import Path
from collections import deque


HEADERS = {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
}

class InsertMiddleware:
    def __init__(self, app, max_concurrent=50):
        self.app = app
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue = deque()

    async def process_request(self, scope, receive, send):
        async with self.semaphore:

            log = f"Received request {scope['request']['uuid']} in queue {scope['request']['time_in_queue']} seconds"
            logging.info(log)

            queue = asyncio.Queue()

            async def message_poller(sentinel, handler_task):
                nonlocal queue
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        handler_task.cancel()
                        return sentinel
                    await queue.put(message)

            sentinel = object()
            handler_task = asyncio.create_task(self.app(scope, queue.get, send))
            asyncio.create_task(message_poller(sentinel, handler_task))

            try:
                await handler_task
                
                if 'after_time_taken' in scope['request']:
                    before_time_taken = scope['request']['before_time_taken']
                    after_time_taken = scope['request']['after_time_taken']
                    total_page = scope['request']['total_page']

                    time_taken = after_time_taken - before_time_taken
                    page_per_second = total_page / time_taken

                    s = f"Complete {scope['request']['uuid']}, time taken {time_taken} seconds, Page per second {page_per_second}"
                    logging.info(s)
                
                if 'time_max_tokens' in scope['request']:
                    time_taken_first_token = scope['request']['time_first_token'] - \
                        scope['request']['after_queue']
                    time_taken_max_tokens = scope['request']['time_max_tokens'] - \
                        scope['request']['time_first_token']
                    tps = scope['request']['total_tokens'] / time_taken_max_tokens

                    s = f"Complete {scope['request']['uuid']}, time first token {time_taken_first_token} seconds, time taken {time_taken_max_tokens} seconds, TPS {tps}"
                    logging.info(s)

            except asyncio.CancelledError:
                logging.warning(f"Cancelling {scope['request']['uuid']} due to disconnect")
            finally:

                if 'cache' in scope and scope['cache'] is not None:
                    cleanup_cache(scope['cache'])
                    scope.pop('cache', None)

                torch.cuda.empty_cache()
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        scope['request'] = {
            'uuid': str(uuid.uuid4()),
            'before_queue': time.time()
        }

        if self.semaphore.locked():
            logging.debug(f"{scope['request']['uuid']} waiting for queue.")
            future = asyncio.Future()
            self.queue.append(future)
            await future

        scope['request']['after_queue'] = time.time()
        scope['request']['time_in_queue'] = scope['request']['after_queue'] - \
            scope['request']['before_queue']

        await self.process_request(scope, receive, send)

        if self.queue:
            next_request = self.queue.popleft()
            next_request.set_result(None)

app = FastAPI()

app.add_middleware(InsertMiddleware, max_concurrent=args.max_concurrent)

@app.get('/')
async def hello_world():
    return {'hello': 'world'}

@app.post('/doc_layout')
async def doc_layout(
    file: bytes = File(), 
    iou_threshold: float = Form(0.45),
    ratio_x: float = Form(2.0),
    ratio_y: float = Form(2.0),
    request: Request = None,
):
    """
    Support pdf file, one file multiple pages.
    """

    with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_file:
        temp_file.write(file)

        r = await doc_layout_predict(
            temp_file, 
            iou_threshold = iou_threshold,
            ratio_x = ratio_x,
            ratio_y = ratio_y,
            request = request
        )
        return r

@app.post('/ocr')
async def ocr(
    image: bytes = File(), 
    max_tokens: int = Form(4096),
    stream: bool = Form(False),
    request: Request = None,
):
    generator = ocr_predict(image, max_tokens = max_tokens, stream = stream, request = request)
    r = await generator
    if stream:
        return EventSourceResponse(r, headers=HEADERS)
    else:
        return r


if args.dynamic_batching:
    @app.on_event("startup")
    async def startup_event():
        app.state.background_doc_layout_step = asyncio.create_task(doc_layout_step())
        app.state.background_ocr_prefill = asyncio.create_task(ocr_prefill())
        app.state.background_ocr_step = asyncio.create_task(ocr_step())

    @app.on_event("shutdown")
    async def shutdown_event():
        app.state.background_doc_layout_step.cancel()
        app.state.background_ocr_prefill.cancel()
        app.state.background_ocr_step.cancel()
        try:
            await app.state.background_doc_layout_step
            await app.state.background_ocr_prefill
            await app.state.background_ocr_step
        except asyncio.CancelledError:
            pass

if args.hotload:
    logging.info('hotloading the model')
    doc_layout_load_model()
    ocr_load_model()

if __name__ == "__main__":
    uvicorn.run(
        'dynamicbatch_ragpipeline.main:app',
        host=args.host,
        port=args.port,
        log_level=args.loglevel.lower(),
        access_log=True,
        reload=args.reload,
    )