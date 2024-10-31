# dynamic-batch-RAG-pipeline

Dynamic batching for Document Layout and OCR, suitable for RAG.

1. Dynamic batching for SOTA Document Layout and OCR, suitable to serve better concurrency.
2. Continuous batching for Causal based OCR models.
3. Can serve user defined max concurrency.
4. Disconnected signal, so this is to ensure early stop for continuous batching.

**Yeah I know, repository name kinda sucks**.

## Available models

### Document Layout

1. https://github.com/opendatalab/DocLayout-YOLO

### OCR

1. https://huggingface.co/stepfun-ai/GOT-OCR2_0

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

```bash
curl -X 'POST' \
  'http://localhost:7088/doc_layout' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@stress-test/2310.01889v4.pdf;type=application/pdf' \
  -F 'iou_threshold=0.45'
```

Checkout [notebook/document-layout.ipynb](notebook/document-layout.ipynb).

<img src="notebook/doc-layout.png" height="50%">

#### Example request OCR

```bash
curl -X 'POST' \
  'http://localhost:7088/ocr' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@stress-test/table2.png;type=image/png' \
  -F 'max_tokens=4096' \
  -F 'stream=false'
```

**Because the backend is a continuous batching, so we support streaming**.

Checkout [notebook/ocr.ipynb](notebook/ocr.ipynb).

```
[{'result': ' \\\\ \\(\\underline{\\text { Ring Attention with Blockwise }}\\)\n \\\\ Transformers for Near-Infinite Context\n}'},
 {'result': ' \\\\ Hao Liu, Matei Zaharia, Pieter Abbeel\n UC Berkeley\n hao.liu@cs.berkeley.edu\n}'},
 {'result': '\\text{Hao Liu, Matei Zaharia, Pieter Abbeel}'},
 {'result': '\\begin{tikzpicture}\n\\draw[thick, ->] (-5.1, 1) -- (5.5, 1);\n\\foreach \\x in {-4,6,8,10,11,12,13,14,15}\n\\draw (\\x,1.1) -- (\\x,0.1) node[below,color=red] {$\\x$};\n\\foreach \\y in {2,4}\n\\draw (0.1,\\y) -- (-0.1,\\y) node[below,color=red] {$\\y$};\n\\draw[line width=2.4pt] (0,1) -- (-5.1,1);\n\\draw[line width=2.4pt] (0,1) -- (-5.1,1);\n\\draw[line width=2.4pt] (0,1) -- (-5.1,1);\n\\end{tikzpicture}'},
 {'result': '\\text{Abstract}'},
 {'result': ' Transformers have emerged as the architecture of choice for many state-of-the-art \\(\\mathrm{AI}\\) models,\n showcasing exceptional performance across a wide range of AI applications. However, the\n memory demands imposed by Transformers limit their ability to handle long sequences, thereby\n posing challenges in utilizing videos, actions, and other long-form sequences and modalities\n in complex environments. We present a novel approach, Ring Attention with Blockwise\n Transformers (Ring Attention), which leverages blockwise computation of self-attention and\nfeedforward to distribute long sequences across multiple devices while fully overlapping the\ncommunication of key-value blocks with the computation of blockwise attention. Our approach\nenables training and inference of sequences that are up to device count times longer than\nthose achievable by prior memory-efficient Transformers, without resorting to approximations\nor incurring additional communication and computation overheads. Extensive experiments\non language modeling and reinforcement learning tasks demonstrate the effectiveness of our\napproach in allowing millions of tokens context size and improving performance. \\({ }^{1}\\).'},
 {'result': '\\text{1 Introduction}'},
 {'result': " Transformers [37] have become the backbone of many state-of-the-art \\(\\mathrm{AI}\\) systems that have demon\nstrated impressive performance across a wide range of AI problems. Transformers achieve this\n success through their architecture design that uses self-attention and position-wise feedforward\n mechanisms. However, scaling up the context length of Transformers is a challenge [29], since the\n inherited architecture design of Transformers, i.e. the self-attention has memory cost quadratic in the\n input sequence length, which makes it challenging to scale to longer input sequences. Large context\n Transformers are essential for tackling a diverse array of AI challenges, ranging from processing\n books and high-resolution images to analyzing long videos and complex codebases. They excel at\n extracting information from the interconnected web and hyperlinked content, and are crucial for\n handling complex scientific experiment data. There have been emerging use cases of language models\n with significantly expanded context than before: GPT-3.5 [32] with context length 16K, GPT-4 [29]\n with context length 32k, MosaicML's MPT [25] with context length 65k, and Anthropic's Claude [1]\n with context length 100k."},
 {'result': " Driven by the significance, there has been surging research interests in reducing memory cost. \\({ }^{2}\\)\n One line of research leverages the observation that the softmax matrix in self-attention can be\n computed without materializing the full matrix [24] which has led to the development of blockwise\n computation of self-attention and feedforward [30, 9, 23] without making approximations. Despite\n the reduced memory, a significant challenge still arises from storing the output of each layer. This\n necessity arises from self-attention's inherent nature, involving interactions among all elements\n (n to \\(\\mathrm{n}\\) interactions). The subsequent layer's self-attention relies on accessing all of the prior\n layer's outputs. Failing to do so would increase computational costs cubically, as every output\n must be recomputed for each sequence element, rendering it impractical for longer sequences."},
 {'result': '\\text{Code: https://github.com/lhao499/llm_large_context}'},
 {'result': '\\text{Preprint.}'}]
```

## [Stress test](stress-test)

### Document layout

Rate of 10 users per second, total requests up to 100 users for 60 seconds on a RTX 3090 Ti,

#### Non-dynamic batching

![alt text](stress-test/doc_layout_without_dynamic.png)

#### Dynamic batching

![alt text](stress-test/doc_layout_dynamic.png)

### OCR

Rate of 5 users per second, total requests up to 50 users for 60 seconds,

#### Continuous batching

![alt text](stress-test/ocr.png)
