from locust import HttpUser, task
from locust import events
import itertools
import time

"""
Make sure already running this,

CUDA_VISIBLE_DEVICES=0 \
python3 -m dynamicbatch_ragpipeline.main \
--host 0.0.0.0 --port 7088 \
--dynamic-batching false
"""

class HelloWorldUser(HttpUser):

    host = "http://127.0.0.1:7088"

    @task
    def hello_world(self):

        files = {
            'file': ('2310.01889v4.pdf', open('2310.01889v4.pdf', 'rb'), 'application/pdf'),
        }
        r = self.client.post('/doc_layout', files=files)