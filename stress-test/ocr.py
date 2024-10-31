from locust import HttpUser, task
from locust import events
import itertools
import time

"""
Make sure already running this,

CUDA_VISIBLE_DEVICES=0 \
python3.10 -m dynamicbatch_ragpipeline.main \
--host 0.0.0.0 --port 7088 \
--dynamic-batching true \
--dynamic-batching-batch-size 32
"""

class HelloWorldUser(HttpUser):

    host = "http://127.0.0.1:7088"

    @task
    def hello_world(self):

        files = {
            'image': ('table1.png', open('table1.png', 'rb'), 'image/png'),
        }
        r = self.client.post('/ocr', files=files)