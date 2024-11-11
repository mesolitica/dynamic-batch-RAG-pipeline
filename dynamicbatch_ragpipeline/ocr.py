from transformers import AutoModel, AutoTokenizer
from sse_starlette import ServerSentEvent
from dynamicbatch_ragpipeline import ocr_utils
from dynamicbatch_ragpipeline.env import args
from dynamicbatch_ragpipeline.function import efficient_attention_mask
from dynamicbatch_ragpipeline.cache import (
    DynamicLengthDecoderCache,
    StaticLengthDecoderCache,
)
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
global_cache = None

torch_dtype = torch.bfloat16

device = 'cpu'
if args.accelerator_type == 'cuda':
    if not torch.cuda.is_available():
        logging.warning('CUDA is not available, fallback to CPU.')
    else:
        device = 'cuda'

prefill_queue = asyncio.Queue()
step_queue = asyncio.Queue()

def load_model():
    global model, tokenizer, global_cache

    if args.model_ocr == 'got_ocr2_0':
        tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
        model = AutoModel.from_pretrained(
            'ucaslcl/GOT-OCR2_0', 
            trust_remote_code=True,
            torch_dtype = torch_dtype,
            attn_implementation = 'sdpa'
        )
        model = model.eval().to(device)

        if args.static_cache:
            logging.info('initiate static cache')
            global_cache = StaticLengthDecoderCache(
                batch_size = args.dynamic_batching_ocr_batch_size, 
                max_length = args.static_cache_max_length,
                device = device,
                head_size = model.config.num_attention_heads,
                dim_size = model.config.hidden_size // model.config.num_attention_heads,
                num_hidden_layers = model.config.num_hidden_layers,
                dtype = model.dtype,
            )
        else:
            global_cache = DynamicLengthDecoderCache()

async def prefill():
    need_sleep = True
    while True:
        if need_sleep:
            await asyncio.sleep(args.dynamic_batching_microsleep)
        
        try:
            batch = []
            while not prefill_queue.empty():
                try:
                    if len(batch) + global_cache.batch_size() >= args.dynamic_batching_ocr_batch_size:
                        need_sleep = False
                        break
                    request = await asyncio.wait_for(prefill_queue.get(), timeout=1e-9)
                    batch.append(request)
                    need_sleep = True
                except asyncio.TimeoutError:
                    break

            if not len(batch):
                continue

            futures = [batch[i][0] for i in range(len(batch))]
            input_img = [batch[i][1] for i in range(len(batch))]
            modes = [batch[i][3] for i in range(len(batch))]
            uuids = [batch[i][4] for i in range(len(batch))]

            prompts = []
            for f in modes:
                if f == 'format':
                    qs = 'OCR with format: '
                else:
                    qs = 'OCR: '

                qs = ocr_utils.qs + qs
                conv = ocr_utils.conv_mpt.copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                prompts.append(prompt)

            images = []
            for i in range(len(input_img)):
                image = Image.open(BytesIO(input_img[i])).convert('RGB')
                image_tensor = ocr_utils.image_processor_high(image).unsqueeze(0).type(model.dtype).to(device)
                images.append(image_tensor)
            
            input_ids = tokenizer(prompts, return_tensors = 'pt', padding = 'longest')
            input_ids.pop('token_type_ids', None)
            lengths = input_ids['attention_mask'].sum(axis = 1)
            for k in input_ids.keys():
                input_ids[k] = input_ids[k].to(device)

            with torch.no_grad():
                out = model(
                    **input_ids,
                    images = images,
                    past_key_values = None,
                    use_cache = True,
                    return_dict = False,
                )
            out_logits = out[0]
            out_caches = out[1]

            cache_exists = len(global_cache.key_cache)
            
            for k in range(len(out_caches)):
                
                key_cache = {}
                value_cache = {}
                for i in range(len(batch)):
                    key_cache[uuids[i]] = out_caches[k][0][i: i + 1]
                    value_cache[uuids[i]] = out_caches[k][1][i: i + 1]
                
                if cache_exists:
                    global_cache.key_cache[k].update(key_cache)
                    global_cache.value_cache[k].update(value_cache)
                else:
                    global_cache.key_cache.append(key_cache)
                    global_cache.value_cache.append(value_cache)
            
            for i in range(len(futures)):
                futures[i].set_result((out_logits[i, -1:],))
            
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
                    if len(batch) + global_cache.batch_size() >= args.dynamic_batching_ocr_batch_size:
                        need_sleep = False
                        break
                    request = await asyncio.wait_for(step_queue.get(), timeout=1e-9)
                    batch.append(request)
                    need_sleep = True
                except asyncio.TimeoutError:
                    break

            if not len(batch):
                continue

            futures = [batch[i][0] for i in range(len(batch))]
            inputs = [batch[i][1] for i in range(len(batch))]
            lengths = [batch[i][2] for i in range(len(batch))]
            uuids = [batch[i][4] for i in range(len(batch))]

            sorted_indices = sorted(range(len(uuids)), key=lambda i: uuids[i])
            inputs = [inputs[i] for i in sorted_indices]
            lengths = [lengths[i] for i in sorted_indices]

            max_len_lengths = max(lengths)
            with torch.no_grad():
                inputs = torch.concat(inputs, dim=0)
                attention_mask = efficient_attention_mask(
                    batch_size=len(lengths),
                    max_len=max_len_lengths,
                    lengths=lengths,
                    device=device,
                    dtype=torch_dtype,
                )
                position_ids = torch.tensor([[l - 1 for l in lengths]]).T.to(device)
                out = model(
                    inputs,
                    images = None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=global_cache,
                    use_cache=True,
                    return_dict=False
                )

            out_logits = out[0]

            for i in range(len(futures)):
                futures[i].set_result((out_logits[i, -1:],))
        
        except Exception as e:
            logging.error(f'error in step {e}')
            try:
                futures = [batch[i][0] for i in range(len(batch))]
                for i in range(len(futures)):
                    if not futures[i].done():
                        futures[i].set_exception(e)
            except:
                pass

async def streaming(image, mode, max_tokens, request):

    cache = None
    length = None
    inputs = image
    uuid = request.scope['request']['uuid']

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
                await q.put((future, inputs, l, mode, uuid))
                out = await future
                
                logits = out[0]
                
                if length is None:
                    length = global_cache.key_cache[0][uuid].shape[2]

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
        
        finally:
            for i in range(len(global_cache.key_cache)):
                global_cache.key_cache[i].pop(request.scope['request']['uuid'], None)
                global_cache.value_cache[i].pop(request.scope['request']['uuid'], None)

async def predict(image, mode = 'format', max_tokens = 4096, stream = False, request = None):
    if model is None:
        load_model()

    func = streaming(image=image, mode=mode, max_tokens=max_tokens, request=request)
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

