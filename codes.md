
## Image Object Detection with Amazon Rekognition

<p align=center>
<img src=./rk.png width=600>
</p>

```python
import json
import boto3

def lambda_handler(event, context):

    bucket_name = "whizlabs.rekognition.23"
    image_obj_name = "rose_flower.jpeg"

    try:
        rkClient = boto3.client("rekognition", region_name="us-east-1")
        try:
            rkResponse = rkClient.detect_labels(
                Image={
                    'S3Object': {
                        'Bucket': bucket_name,
                        'Name': image_obj_name
                    }
                },
            )
            print(rkResponse['Labels'])
            return rkResponse['Labels']
        except Exception as e:
            print("Get labels failed because ", e)
    except Exception as e:
        print("Client connection to Rekognition failed because ", e)
```

## Real Time Data Streaming System

<p align=center>
<img src=./kinesis.png width=600>
</p>

Case Study
1. We have an application which uploads text files to S3 Bucket.
2. Whenever a file is uploaded to the S3 Bucket, it’s going to trigger a lambda function.
3. The lambda function is a data producer, which reads the content from the S3 Bucket and then pushes the data to the Kinesis data stream.
4. We have two consumers which consume the data from the stream.
5. The consumers can do many things with the data.
6. Suppose the consumer can read the data and send an email to the clients with the information or the data can be published into social media platforms or the data can be saved into the database.

