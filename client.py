import os
import io
import boto3
import json
import csv

ENDPOINT_NAME = 'bikeshare-sagemaker-regression-v1'
runtime= boto3.client('runtime.sagemaker')

# Convert categorical date field
observation = "231,2011-08-19,3,0,8,0,5,1,2,0.685,0.633221,0.722917,0.139308,797,3356"
observation = observation.replace("-","")

data = {"data":observation}
actual = {"data":"4153"}
payload = data['data']
# print(payload)

response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                    ContentType='text/csv',
                                    Body=payload)
# print(response)

result = json.loads(response['Body'].read().decode())
print('prediction: ', result['predictions'][0]['score'], '\t\tactual: ', str(actual['data']))
