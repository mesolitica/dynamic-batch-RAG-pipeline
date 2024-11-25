# Stress test

## Document layout on RTX 3090 Ti

Rate of 10 users per second, total requests up to 100 users for 60 seconds,

```bash
locust -f doc_layout.py -P 7001 -H http://localhost:7088 -r 10 -u 100 -t 60
```

![alt text](doc_layout.png)

## OCR on RTX 3090 Ti

Rate of 5 users per second, total requests up to 50 users for 60 seconds,

```bash
locust -f ocr.py -P 7001 -H http://localhost:7088 -r 5 -u 50 -t 60
```

### Continuous batching

![alt text](ocr.png)