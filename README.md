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
usage: main.py [-h] [--host HOST] [--port PORT] [--loglevel LOGLEVEL] [--model-doc-layout MODEL_DOC_LAYOUT]
               [--dynamic-batching-microsleep DYNAMIC_BATCHING_MICROSLEEP] [--dynamic-batching-batch-size DYNAMIC_BATCHING_BATCH_SIZE]
               [--accelerator-type ACCELERATOR_TYPE] [--max-concurrent MAX_CONCURRENT]

Configuration parser

options:
  -h, --help            show this help message and exit
  --host HOST           host name to host the app (default: 0.0.0.0, env: HOSTNAME)
  --port PORT           port to host the app (default: 7088, env: PORT)
  --loglevel LOGLEVEL   Logging level (default: INFO, env: LOGLEVEL)
  --model-doc-layout MODEL_DOC_LAYOUT
                        Model type (default: yolo10, env: MODEL_DOC_LAYOUT)
  --dynamic-batching-microsleep DYNAMIC_BATCHING_MICROSLEEP
                        microsleep to group dynamic batching, 1 / 1e-4 = 10k steps for second (default: 0.0001, env:
                        DYNAMIC_BATCHING_MICROSLEEP)
  --dynamic-batching-batch-size DYNAMIC_BATCHING_BATCH_SIZE
                        maximum of batch size during dynamic batching (default: 50, env: DYNAMIC_BATCHING_BATCH_SIZE)
  --accelerator-type ACCELERATOR_TYPE
                        Accelerator type (default: cuda, env: ACCELERATOR_TYPE)
  --max-concurrent MAX_CONCURRENT
                        Maximum concurrent requests (default: 100, env: MAX_CONCURRENT)
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

Output,

```
[{"classes":["plain text","plain text","plain text","title","title","abandon","title","abandon","abandon","plain text","plain text"],"coordinates":[{"x_min":350,"y_min":804,"x_max":1487,"y_max":1234},{"x_min":321,"y_min":1352,"x_max":1516,"y_max":1781},{"x_min":322,"y_min":1795,"x_max":1516,"y_max":2091},{"x_min":482,"y_min":293,"x_max":1356,"y_max":410},{"x_min":850,"y_min":727,"x_max":987,"y_max":767},{"x_min":362,"y_min":2108,"x_max":917,"y_max":2137},{"x_min":323,"y_min":1280,"x_max":575,"y_max":1321},{"x_min":46,"y_min":621,"x_max":112,"y_max":1682},{"x_min":321,"y_min":2197,"x_max":417,"y_max":2227},{"x_min":668,"y_min":526,"x_max":1169,"y_max":642},{"x_min":668,"y_min":528,"x_max":1167,"y_max":563}]},{"classes":["plain text","plain text","plain text","plain text","plain text","figure","plain text","plain text","title","isolate_formula","figure_caption","abandon"],"coordinates":[{"x_min":322,"y_min":218,"x_max":894,"y_max":581},{"x_min":320,"y_min":956,"x_max":1516,"y_max":1416},{"x_min":320,"y_min":1429,"x_max":1516,"y_max":1661},{"x_min":321,"y_min":595,"x_max":894,"y_max":956},{"x_min":321,"y_min":1675,"x_max":1517,"y_max":1809},{"x_min":914,"y_min":223,"x_max":1512,"y_max":556},{"x_min":319,"y_min":2100,"x_max":1518,"y_max":2169},{"x_min":320,"y_min":1915,"x_max":1519,"y_max":1985},{"x_min":322,"y_min":1841,"x_max":916,"y_max":1882},{"x_min":657,"y_min":2004,"x_max":1182,"y_max":2083},{"x_min":914,"y_min":563,"x_max":1520,"y_max":926},{"x_min":907,"y_min":2226,"x_max":927,"y_max":2252}]},{"classes":["plain text","plain text","plain text","plain text","title","plain text","isolate_formula","plain text","abandon"],"coordinates":[{"x_min":320,"y_min":1085,"x_max":1517,"y_max":1483},{"x_min":320,"y_min":1494,"x_max":1518,"y_max":2087},{"x_min":321,"y_min":685,"x_max":1515,"y_max":983},{"x_min":321,"y_min":342,"x_max":1518,"y_max":671},{"x_min":322,"y_min":1014,"x_max":1194,"y_max":1056},{"x_min":321,"y_min":2100,"x_max":1516,"y_max":2168},{"x_min":672,"y_min":271,"x_max":1160,"y_max":310},{"x_min":321,"y_min":218,"x_max":943,"y_max":252},{"x_min":908,"y_min":2227,"x_max":927,"y_max":2252}]},{"classes":["plain text","figure","figure_caption","abandon","figure_caption"],"coordinates":[{"x_min":320,"y_min":1805,"x_max":1516,"y_max":2168},{"x_min":336,"y_min":215,"x_max":1498,"y_max":1351},{"x_min":320,"y_min":1388,"x_max":1516,"y_max":1751},{"x_min":908,"y_min":2228,"x_max":926,"y_max":2251},{"x_min":870,"y_min":1358,"x_max":1092,"y_max":1384}]},{"classes":["plain text","plain text","plain text","table","table","plain text","table_caption","table_caption","abandon"],"coordinates":[{"x_min":320,"y_min":1345,"x_max":1515,"y_max":1678},{"x_min":320,"y_min":1968,"x_max":1516,"y_max":2169},{"x_min":320,"y_min":1690,"x_max":1516,"y_max":1955},{"x_min":404,"y_min":857,"x_max":1420,"y_max":1174},{"x_min":479,"y_min":427,"x_max":1350,"y_max":680},{"x_min":320,"y_min":1231,"x_max":1517,"y_max":1332},{"x_min":319,"y_min":704,"x_max":1517,"y_max":840},{"x_min":319,"y_min":208,"x_max":1518,"y_max":408},{"x_min":907,"y_min":2227,"x_max":927,"y_max":2252}]},{"classes":["plain text","plain text","plain text","plain text","plain text","plain text","title","title","title","plain text","abandon","plain text"],"coordinates":[{"x_min":321,"y_min":1835,"x_max":1516,"y_max":2167},{"x_min":321,"y_min":1531,"x_max":1518,"y_max":1731},{"x_min":321,"y_min":1249,"x_max":1517,"y_max":1415},{"x_min":322,"y_min":1101,"x_max":1516,"y_max":1235},{"x_min":322,"y_min":1020,"x_max":1516,"y_max":1087},{"x_min":322,"y_min":937,"x_max":1515,"y_max":1005},{"x_min":322,"y_min":1458,"x_max":493,"y_max":1497},{"x_min":322,"y_min":1768,"x_max":768,"y_max":1805},{"x_min":322,"y_min":863,"x_max":490,"y_max":906},{"x_min":329,"y_min":251,"x_max":1516,"y_max":791},{"x_min":907,"y_min":2228,"x_max":927,"y_max":2252},{"x_min":320,"y_min":218,"x_max":1196,"y_max":254}]},{"classes":["plain text","table","plain text","plain text","title","table_caption","abandon"],"coordinates":[{"x_min":320,"y_min":1723,"x_max":1516,"y_max":2054},{"x_min":349,"y_min":425,"x_max":1479,"y_max":1297},{"x_min":321,"y_min":1325,"x_max":1516,"y_max":1622},{"x_min":321,"y_min":2067,"x_max":1516,"y_max":2167},{"x_min":322,"y_min":1656,"x_max":842,"y_max":1694},{"x_min":319,"y_min":208,"x_max":1517,"y_max":408},{"x_min":908,"y_min":2226,"x_max":927,"y_max":2252}]},{"classes":["plain text","table","plain text","title","table_caption","figure","figure_caption","abandon"],"coordinates":[{"x_min":320,"y_min":1329,"x_max":1517,"y_max":1696},{"x_min":353,"y_min":1839,"x_max":1482,"y_max":2162},{"x_min":321,"y_min":1135,"x_max":1516,"y_max":1236},{"x_min":322,"y_min":1266,"x_max":892,"y_max":1303},{"x_min":320,"y_min":1719,"x_max":1516,"y_max":1821},{"x_min":321,"y_min":336,"x_max":1506,"y_max":1087},{"x_min":318,"y_min":208,"x_max":1519,"y_max":311},{"x_min":907,"y_min":2227,"x_max":927,"y_max":2252}]},{"classes":["plain text","plain text","plain text","figure","figure_caption","title","title","abandon"],"coordinates":[{"x_min":321,"y_min":1510,"x_max":1517,"y_max":2169},{"x_min":321,"y_min":1009,"x_max":1516,"y_max":1405},{"x_min":321,"y_min":650,"x_max":1516,"y_max":916},{"x_min":325,"y_min":177,"x_max":1506,"y_max":578},{"x_min":444,"y_min":595,"x_max":1389,"y_max":631},{"x_min":322,"y_min":1439,"x_max":593,"y_max":1479},{"x_min":322,"y_min":946,"x_max":770,"y_max":980},{"x_min":907,"y_min":2227,"x_max":926,"y_max":2251}]},{"classes":["plain text","plain text","plain text","title","title","title","abandon","plain text","plain text","plain text","plain text","plain text","plain text"],"coordinates":[{"x_min":321,"y_min":787,"x_max":1516,"y_max":1150},{"x_min":321,"y_min":217,"x_max":1517,"y_max":679},{"x_min":321,"y_min":1256,"x_max":1517,"y_max":1487},{"x_min":323,"y_min":1185,"x_max":609,"y_max":1226},{"x_min":322,"y_min":714,"x_max":554,"y_max":755},{"x_min":322,"y_min":1524,"x_max":494,"y_max":1564},{"x_min":904,"y_min":2227,"x_max":935,"y_max":2253},{"x_min":333,"y_min":1593,"x_max":1520,"y_max":2171},{"x_min":334,"y_min":2100,"x_max":1517,"y_max":2168},{"x_min":337,"y_min":1933,"x_max":1517,"y_max":2066},{"x_min":381,"y_min":1596,"x_max":1493,"y_max":1666},{"x_min":337,"y_min":1796,"x_max":1516,"y_max":1901},{"x_min":337,"y_min":1697,"x_max":1517,"y_max":1765}]},{"classes":["abandon","plain text","plain text","plain text","plain text","plain text","plain text","plain text","plain text","plain text","plain text","plain text","plain text","plain text","plain text","plain text","plain text","plain text"],"coordinates":[{"x_min":903,"y_min":2226,"x_max":932,"y_max":2253},{"x_min":321,"y_min":1196,"x_max":1518,"y_max":1329},{"x_min":321,"y_min":1352,"x_max":1516,"y_max":1454},{"x_min":321,"y_min":1015,"x_max":1515,"y_max":1084},{"x_min":320,"y_min":1107,"x_max":1516,"y_max":1174},{"x_min":320,"y_min":835,"x_max":1517,"y_max":903},{"x_min":321,"y_min":1722,"x_max":1516,"y_max":1823},{"x_min":320,"y_min":1846,"x_max":1516,"y_max":1948},{"x_min":320,"y_min":1476,"x_max":1515,"y_max":1577},{"x_min":321,"y_min":1600,"x_max":1517,"y_max":1700},{"x_min":317,"y_min":203,"x_max":1521,"y_max":2169},{"x_min":322,"y_min":1968,"x_max":1518,"y_max":2168},{"x_min":322,"y_min":712,"x_max":1517,"y_max":813},{"x_min":322,"y_min":925,"x_max":1514,"y_max":994},{"x_min":334,"y_min":589,"x_max":1516,"y_max":689},{"x_min":336,"y_min":341,"x_max":1517,"y_max":444},{"x_min":337,"y_min":464,"x_max":1517,"y_max":567},{"x_min":335,"y_min":218,"x_max":1518,"y_max":320}]},{"classes":["abandon","plain text","plain text","plain text","plain text","plain text","plain text","plain text","plain text","plain text","plain text","plain text","plain text","plain text","plain text","plain text","plain text","plain text"],"coordinates":[{"x_min":903,"y_min":2226,"x_max":934,"y_max":2253},{"x_min":321,"y_min":1062,"x_max":851,"y_max":1097},{"x_min":321,"y_min":1222,"x_max":1517,"y_max":1325},{"x_min":321,"y_min":1351,"x_max":1517,"y_max":1520},{"x_min":320,"y_min":931,"x_max":1518,"y_max":1032},{"x_min":318,"y_min":1126,"x_max":1517,"y_max":1194},{"x_min":320,"y_min":770,"x_max":1516,"y_max":903},{"x_min":320,"y_min":1644,"x_max":1515,"y_max":1747},{"x_min":321,"y_min":1775,"x_max":1517,"y_max":1906},{"x_min":320,"y_min":1937,"x_max":1518,"y_max":2039},{"x_min":319,"y_min":2067,"x_max":1517,"y_max":2168},{"x_min":320,"y_min":1547,"x_max":1515,"y_max":1617},{"x_min":320,"y_min":218,"x_max":1519,"y_max":288},{"x_min":317,"y_min":201,"x_max":1521,"y_max":2193},{"x_min":319,"y_min":316,"x_max":1517,"y_max":384},{"x_min":320,"y_min":509,"x_max":1517,"y_max":580},{"x_min":321,"y_min":606,"x_max":1518,"y_max":740},{"x_min":380,"y_min":412,"x_max":1493,"y_max":481}]},{"classes":["abandon","plain text","plain text"],"coordinates":[{"x_min":903,"y_min":2227,"x_max":934,"y_max":2253},{"x_min":319,"y_min":406,"x_max":1517,"y_max":508},{"x_min":320,"y_min":218,"x_max":1519,"y_max":385}]},{"classes":["plain text","plain text","plain text","plain text","plain text","plain text","title","title","title","title","title","title","abandon"],"coordinates":[{"x_min":321,"y_min":1838,"x_max":1515,"y_max":2167},{"x_min":322,"y_min":1504,"x_max":1516,"y_max":1737},{"x_min":321,"y_min":464,"x_max":1516,"y_max":827},{"x_min":321,"y_min":993,"x_max":1517,"y_max":1193},{"x_min":322,"y_min":1281,"x_max":1516,"y_max":1415},{"x_min":322,"y_min":285,"x_max":1515,"y_max":451},{"x_min":322,"y_min":857,"x_max":684,"y_max":899},{"x_min":323,"y_min":1219,"x_max":645,"y_max":1254},{"x_min":322,"y_min":1442,"x_max":749,"y_max":1478},{"x_min":321,"y_min":930,"x_max":758,"y_max":966},{"x_min":322,"y_min":1767,"x_max":733,"y_max":1810},{"x_min":323,"y_min":213,"x_max":469,"y_max":254},{"x_min":903,"y_min":2227,"x_max":933,"y_max":2252}]},{"classes":["abandon","plain text","plain text"],"coordinates":[{"x_min":903,"y_min":2226,"x_max":934,"y_max":2253},{"x_min":291,"y_min":204,"x_max":1669,"y_max":2149},{"x_min":316,"y_min":2144,"x_max":1516,"y_max":2213}]},{"classes":["plain text","plain text","plain text","plain text","figure_caption","title","abandon","figure"],"coordinates":[{"x_min":321,"y_min":388,"x_max":1516,"y_max":655},{"x_min":321,"y_min":1587,"x_max":1517,"y_max":1789},{"x_min":321,"y_min":1802,"x_max":1517,"y_max":1936},{"x_min":320,"y_min":217,"x_max":1516,"y_max":287},{"x_min":320,"y_min":1459,"x_max":1518,"y_max":1556},{"x_min":321,"y_min":316,"x_max":1001,"y_max":360},{"x_min":903,"y_min":2227,"x_max":934,"y_max":2253},{"x_min":347,"y_min":692,"x_max":1486,"y_max":1448}]}]
```