from dynamicbatch_ragpipeline.env import args
from dynamicbatch_ragpipeline.doc_layout import step, load_model, predict
from fastapi import FastAPI, Request
from fastapi import File, Form, UploadFile
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
            except asyncio.CancelledError:
                logging.warning(f"Cancelling {scope['request']['uuid']} due to disconnect")
            finally:
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
    return_image: bool = Form(False),
    request: Request = None,
):
    """
    Support pdf file, one file multiple pages.

    If return_image is True, will return JPEG base64.
    """

    with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_file:
        temp_file.write(file)

        r = await predict(
            temp_file, 
            iou_threshold = iou_threshold,
            return_image = return_image,
            request = request
        )
        return r


@app.on_event("startup")
async def startup_event():
    app.state.background_step = asyncio.create_task(step())

@app.on_event("shutdown")
async def shutdown_event():
    app.state.background_step.cancel()
    try:
        await app.state.background_step
    except asyncio.CancelledError:
        pass

if args.hotload:
    logging.info('hotloading the model')
    load_model()

if __name__ == "__main__":
    uvicorn.run(
        'dynamicbatch_ragpipeline.main:app',
        host=args.host,
        port=args.port,
        log_level=args.loglevel.lower(),
        access_log=True,
        reload=args.reload,
    )