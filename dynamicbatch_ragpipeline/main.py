from fastapi import FastAPI, Request
from fastapi import File, Form, UploadFile
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
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
from dynamicbatch_ragpipeline.playwright_utils import (
    to_pdf,
    initialize_browser,
)
from transformers_openai.middleware import InsertMiddleware
from pydantic import BaseModel
import asyncio
import logging
import uvicorn
import tempfile


app = FastAPI()

app.add_middleware(InsertMiddleware, max_concurrent=args.max_concurrent)

@app.get('/')
async def hello_world():
    return {'hello': 'world'}
            
if args.enable_doc_layout:
    logging.info('enabling document layout')

    @app.post('/doc_layout')
    async def doc_layout(
        file: bytes = File(), 
        iou_threshold: float = Form(0.45),
        ratio_x: float = Form(2.0),
        ratio_y: float = Form(2.0),
        request: Request = None,
    ):
        """
        Support pdf file, one file multiple pages. Will return list of images in base64 with list of layouts.
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

    doc_layout_load_model()

    @app.on_event("startup")
    async def startup_event():
        app.state.background_doc_layout_step = asyncio.create_task(doc_layout_step())

    @app.on_event("shutdown")
    async def shutdown_event():
        app.state.background_doc_layout_step.cancel()
        try:
            await app.state.background_doc_layout_step
        except asyncio.CancelledError:
            pass

if args.enable_ocr:
    logging.info('enabling OCR')

    @app.post('/ocr')
    async def ocr(
        image: bytes = File(),
        mode: str = Form('format'),
        max_tokens: int = Form(4096),
        stream: bool = Form(False),
        request: Request = None,
    ):
        """
        Convert image to text using OCR.

        Support 2 modes,

        1. `plain`, plain text.

        2. `format`, will format into latex.

        """
        mode = mode.lower()
        if mode not in {'plain', 'format'}:
            raise HTTPException(status_code=400, detail='mode only support `plain` or `format`.')

        generator = ocr_predict(image, mode = mode, max_tokens = max_tokens, stream = stream, request = request)
        r = await generator
        if stream:
            return EventSourceResponse(r, headers=HEADERS)
        else:
            return r

    ocr_load_model()
    
    @app.on_event("startup")
    async def startup_event():
        app.state.background_ocr_prefill = asyncio.create_task(ocr_prefill())
        app.state.background_ocr_step = asyncio.create_task(ocr_step())

    @app.on_event("shutdown")
    async def shutdown_event():
        app.state.background_ocr_prefill.cancel()
        app.state.background_ocr_step.cancel()
        try:
            await app.state.background_ocr_prefill
            await app.state.background_ocr_step
        except asyncio.CancelledError:
            pass

if args.enable_url_to_pdf:
    logging.info('enabling URL to PDF')

    class URL(BaseModel):
        url: str = 'https://screenresolutiontest.com/screenresolution/'
        viewport_weight: int = 1470
        viewport_height: int = 956

    @app.post('/url_to_pdf')
    async def url_to_pdf(url: URL):
        pdf_file = await to_pdf(**url.dict())
        return StreamingResponse(
            pdf_file,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=mydocument.pdf"}
        )
    
    @app.on_event('startup')
    async def warmup():
        tasks = []
        for index in range(args.playwright_max_concurrency):
            task = asyncio.create_task(initialize_browser(index=index))
            tasks.append(task)

        await asyncio.gather(*tasks)

if __name__ == "__main__":
    uvicorn.run(
        'dynamicbatch_ragpipeline.main:app',
        host=args.host,
        port=args.port,
        log_level=args.loglevel.lower(),
        access_log=True,
        reload=args.reload,
    )