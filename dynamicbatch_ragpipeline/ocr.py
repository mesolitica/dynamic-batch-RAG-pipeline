from transformers import AutoModel, AutoTokenizer
from dynamicbatch_ragpipeline.env import args
import torch
import asyncio

model = None
tokenizer = None

device = 'cpu'
if args.accelerator_type == 'cuda':
    if not torch.cuda.is_available():
        logging.warning('CUDA is not available, fallback to CPU.')
    else:
        device = 'cuda'

prefill_queue = asyncio.Queue()
step_queue = asyncio.Queue()

def load_model():
    global model, tokenizer

    if args.model_ocr == 'got_ocr2_0':
        tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
        model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True)
        model = model.eval().device(device)

async def prefill():
    pass

async def step():
    pass

async def predict(
    image, 
    request = None,
):
    request.scope['request']['before_time_taken'] = time.time()
    
