import requests
import json

headers = {'Content-Type': 'application/json'}

gpt = {"text":"测试测试，这里是测试","speaker":"阿星(粤语)","new":1,"streaming":1}

response = requests.post("http://localhost:9880/",data=json.dumps(gpt),headers=headers)


audio_data = response.content

with open(f"post请求测试.wav","wb") as f:
    f.write(audio_data)