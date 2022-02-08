
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

Lambda: producer
```javascript
const AWS = require('aws-sdk');
AWS.config.update({
    region: 'us-east-1'
})
const s3 = new AWS.S3();
const kinesis = new AWS.Kinesis();

exports.handler = async (event) => {
    console.log(JSON.stringify(event));
    const bucketName = event.Records[0].s3.bucket.name;
    const keyName = event.Records[0].s3.object.key;
    const params = {
        Bucket: bucketName,
        Key: keyName
    }
    await s3.getObject(params).promise().then(async (data) => {
        const dataString = data.Body.toString();
        const payload = {
            data: dataString
        }
        await sendToKinesis(payload, keyName);
    }, error => {
        console.error(error);
    })
};

async function sendToKinesis(payload, partitionKey) {
    const params = {
        Data: JSON.stringify(payload),
        PartitionKey: partitionKey,
        StreamName: 'whiz-data-stream'
    }

    await kinesis.putRecord(params).promise().then(response => {
        console.log(response);
    }, error => {
        console.error(error);
    })
}
```

Lambda: consumer
```javascript
exports.handler = async (event) => {
    console.log(JSON.stringify(event));
    for (const record of event.Records) {
        const data = JSON.parse(Buffer.from(record.kinesis.data, 'base64'));
        console.log('consumer #1', data);
    }
};
```
