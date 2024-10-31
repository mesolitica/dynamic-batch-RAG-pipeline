from transformers import AutoModel, AutoTokenizer
from sse_starlette import ServerSentEvent
from dynamicbatch_ragpipeline import ocr_utils
from dynamicbatch_ragpipeline.env import args
from dynamicbatch_ragpipeline.function import efficient_attention_mask
from dynamicbatch_ragpipeline.cache import DynamicLengthDecoderCache
from datetime import datetime
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import asyncio
import time
import logging
import json

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
        model = AutoModel.from_pretrained(
            'ucaslcl/GOT-OCR2_0', 
            trust_remote_code=True,
            torch_dtype = torch.bfloat16,
            attn_implementation = 'sdpa'
        )
        model = model.eval().to(device)

async def prefill():
    need_sleep = True
    while True:
        if need_sleep:
            await asyncio.sleep(args.dynamic_batching_microsleep)
        
        try:
            batch = []
            while not prefill_queue.empty():
                try:
                    request = await asyncio.wait_for(prefill_queue.get(), timeout=1e-6)
                    batch.append(request)
                    if len(batch) >= args.dynamic_batching_batch_size:
                        need_sleep = False
                        break
                    else:
                        need_sleep = True
                except asyncio.TimeoutError:
                    break

            if not len(batch):
                continue

            futures = [batch[i][0] for i in range(len(batch))]
            input_img = [batch[i][1] for i in range(len(batch))]

            qs = ocr_utils.qs + 'OCR with format: '
            conv = ocr_utils.conv_mpt.copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt], return_tensors = 'pt')

            images = []
            for i in range(len(input_img)):
                image = Image.open(BytesIO(input_img[i])).convert('RGB')
                image_tensor = ocr_utils.image_processor_high(image).unsqueeze(0).type(model.dtype).to(device)
                images.append(image_tensor)
            
            input_ids = inputs['input_ids'].to(device)
            input_ids = input_ids.repeat(len(images), 1)

            out = model(
                input_ids = input_ids, 
                images = images,
                past_key_values = None,
                use_cache = True,
                return_dict = False,
            )
            out_logits = out[0]
            out_caches = out[1]
            
            caches = []
            for i in range(len(batch)):
                cache = []
                for k in range(len(out_caches)):
                    cache_ = [
                        out_caches[k][0][i:i + 1],
                        out_caches[k][1][i:i + 1],
                    ]
                    cache.append(cache_)
                caches.append(cache)
            
            for i in range(len(futures)):
                futures[i].set_result((out_logits[i, -1:], caches[i]))
            
            for k in range(len(out_caches)):
                temp = list(out_caches[k])
                for j in range(len(out_caches[k])):
                    del temp[0]
        
        except Exception as e:
            logging.error(f'error in prefill {e}')
            try:
                futures = [batch[i][0] for i in range(len(batch))]
                for i in range(len(futures)):
                    if not futures[i].done():
                        futures[i].set_exception(e)
            except:
                pass

async def step():
    need_sleep = True
    while True:
        if need_sleep:
            await asyncio.sleep(args.dynamic_batching_microsleep)
        
        try:
            batch = []
            while not step_queue.empty():
                try:
                    request = await asyncio.wait_for(step_queue.get(), timeout=1e-6)
                    batch.append(request)
                    if len(batch) >= args.dynamic_batching_batch_size:
                        need_sleep = False
                        break
                    else:
                        need_sleep = True
                except asyncio.TimeoutError:
                    break

            if not len(batch):
                continue

            futures = [batch[i][0] for i in range(len(batch))]
            inputs = [batch[i][1] for i in range(len(batch))]
            caches = [batch[i][2] for i in range(len(batch))]
            lengths = [batch[i][3] for i in range(len(batch))]

            cache_dtype = caches[0][0][0].dtype
            cache_device = caches[0][0][0].device

            kv_len = [caches[i][0][0].shape[2] for i in range(len(batch))]
            max_len = max(kv_len)
            max_len_lengths = max(lengths)

            cache = DynamicLengthDecoderCache(lengths=lengths)

            len_cache = len(caches[0])
            len_kv = len(caches[0][0])

            for n in range(len_cache):

                key_cache = []
                value_cache = []
                for i in range(len(batch)):
                    key_cache.append(caches[i][n][0])
                    value_cache.append(caches[i][n][1])

                cache.key_cache.append(key_cache)
                cache.value_cache.append(value_cache)
            
            inputs = torch.concat(inputs, dim=0)
            attention_mask = efficient_attention_mask(
                batch_size=len(lengths),
                max_len=max_len_lengths,
                lengths=lengths,
                device=cache_device,
                dtype=cache_dtype,
            )
            position_ids = torch.tensor([[l - 1 for l in lengths]]).T.to(cache_device)

            out = model(
                inputs,
                images = None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=False
            )

            out_logits = out[0]

            caches = []
            for i in range(len(batch)):
                new_cache = []
                for k in range(len(cache)):
                    keys = cache.key_cache[k]
                    values = cache.value_cache[k]
                    v = [keys[i], values[i]]
                    new_cache.append(v)
                caches.append(new_cache)

            for i in range(len(futures)):
                futures[i].set_result((out_logits[i, -1:], caches[i]))

            for k in range(len(cache)):
                temp = list(cache[k])
                for j in range(len(temp)):
                    del temp[0]
        
        except Exception as e:
            logging.error(f'error in step {e}')
            try:
                futures = [batch[i][0] for i in range(len(batch))]
                for i in range(len(futures)):
                    if not futures[i].done():
                        futures[i].set_exception(e)
            except:
                pass

async def streaming(image, max_tokens, request):

    cache = None
    length = None
    inputs = image

    with torch.no_grad():
        try:
            for k in range(max_tokens):

                if k == 0:
                    q = prefill_queue
                    l = length
                else:
                    q = step_queue
                    l = length + k

                future = asyncio.Future()
                await q.put((future, inputs, cache, l))
                out = await future
                
                logits = out[0]
                cache = out[1]
                request.scope['cache'] = cache
                
                if length is None:
                    length = cache[0][0].shape[2]

                idx_next = logits.argmax(-1)
                token = tokenizer.decode(idx_next)

                if k == 0:
                    request.scope['request']['time_first_token'] = time.time()
                
                if token == ocr_utils.stop_str:
                    break

                del logits
                inputs = idx_next.unsqueeze(0)

                data = {
                    'token': token
                }
                yield json.dumps(data)
                await asyncio.sleep(0)
            
            request.scope['request']['time_max_tokens'] = time.time()
            request.scope['request']['total_tokens'] = k

        except asyncio.CancelledError as e:
            logging.warning(f"model step cancelled {request.scope['request']['uuid']}")
            yield ServerSentEvent(**{"data": str(e)})
        
        except Exception as e:
            logging.error(f"model step exception {e} {request.scope['request']['uuid']}")
            yield ServerSentEvent(**{"data": str(e)})

async def predict(image, max_tokens = 4096, stream = False, request = None):
    if model is None:
        load_model()

    func = streaming(image=image, max_tokens=max_tokens, request=request)
    if stream:
        return func
    else:
        tokens = []
        async for data in func:
            if isinstance(data, ServerSentEvent):
                continue
            data = json.loads(data)
            tokens.append(data['token'])
        
        return {
            'result': ''.join(tokens)
        }


    

    
