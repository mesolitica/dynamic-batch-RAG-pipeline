# Stress test

## Document layout on RTX 3090 Ti

Rate of 10 users per second, total requests up to 100 users for 60 seconds,

```bash
locust -f t5_continuous.py -P 7001 -H http://localhost:7088 -r 5 -u 100 -t 60
locust -f doc_layout_without_dynamic -P 7001 -H http://localhost:7088 -r 10 -u 100 -t 60
```

### Non-dynamic batch

![alt text](doc_layout_without_dynamic.png)

### Dynamic batch

![alt text](doc_layout_dynamic.png)