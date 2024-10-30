# dynamic-batch-RAG-pipeline

Dynamic batching for Non-Causal models on Document Layout and OCR, suitable for RAG.

1. Dynamic batch for SOTA Document Layout and OCR, suitable to serve better concurrency.
2. Can serve user defined max concurrency.
3. Disconnected signal, so this is to ensure early stop.

## how to install

Using PIP with git,

```bash
pip3 install git+https://github.com/mesolitica/dynamic-batch-RAG-pipeline
```

Or you can git clone,

```bash
git clone https://github.com/mesolitica/dynamic-batch-RAG-pipeline && cd dynamic-batch-RAG-pipeline
```

## how to

### Supported parameters

```bash
python3 -m dynamicbatch_ragpipeline.main --help
```

```text
usage: main.py [-h] [--host HOST] [--port PORT] [--loglevel LOGLEVEL] [--reload RELOAD] [--model-doc-layout MODEL_DOC_LAYOUT]
               [--dynamic-batching-microsleep DYNAMIC_BATCHING_MICROSLEEP]
               [--dynamic-batching-batch-size DYNAMIC_BATCHING_BATCH_SIZE] [--accelerator-type ACCELERATOR_TYPE]
               [--max-concurrent MAX_CONCURRENT] [--hotload HOTLOAD]

Configuration parser

options:
  -h, --help            show this help message and exit
  --host HOST           host name to host the app (default: 0.0.0.0, env: HOSTNAME)
  --port PORT           port to host the app (default: 7088, env: PORT)
  --loglevel LOGLEVEL   Logging level (default: INFO, env: LOGLEVEL)
  --reload RELOAD       Enable hot loading (default: False, env: RELOAD)
  --model-doc-layout MODEL_DOC_LAYOUT
                        Model type (default: yolo10, env: MODEL_DOC_LAYOUT)
  --dynamic-batching-microsleep DYNAMIC_BATCHING_MICROSLEEP
                        microsleep to group dynamic batching, 1 / 1e-4 = 10k steps for second (default: 0.0001, env:
                        DYNAMIC_BATCHING_MICROSLEEP)
  --dynamic-batching-batch-size DYNAMIC_BATCHING_BATCH_SIZE
                        maximum of batch size during dynamic batching (default: 20, env: DYNAMIC_BATCHING_BATCH_SIZE)
  --accelerator-type ACCELERATOR_TYPE
                        Accelerator type (default: cuda, env: ACCELERATOR_TYPE)
  --max-concurrent MAX_CONCURRENT
                        Maximum concurrent requests (default: 100, env: MAX_CONCURRENT)
  --hotload HOTLOAD     Enable hot loading (default: True, env: HOTLOAD)
```

**We support both args and OS environment**.

### Run

```
python3 -m dynamicbatch_ragpipeline.main \
--host 0.0.0.0 --port 7088
```

#### Example request document layout

```python
curl -X 'POST' \
  'http://100.93.25.29:7088/doc_layout' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@stress-test/2310.01889v4.pdf;type=application/pdf' \
  -F 'iou_threshold=0.45' \
  -F 'return_image=false'
```

## [Stress test](stress-test)

### Document layour

Rate of 10 users per second, total requests up to 100 users for 60 seconds on a RTX 3090 Ti,

### Non-dynamic batch

![alt text](stress-test/doc_layout_without_dynamic.png)

### Dynamic batch

![alt text](stress-test/doc_layout_dynamic.png)