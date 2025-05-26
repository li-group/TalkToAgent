from matplotlib import pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot([2,4,5,6,7], color='yellowgreen')
plt.title('LIME Feature Importance (Mean Absolute Value)')
plt.xlabel('Mean |Importance|')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
fig = plt.gcf()

# %%
# Calling functions with tools
from openai import OpenAI
import os

import os
from dotenv import load_dotenv
import openai

# .env 파일에서 환경변수 불러오기
load_dotenv()

# 환경변수로부터 API 키 가져오기
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
import json

client = OpenAI()

import requests

def get_weather(latitude, longitude):
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = response.json()
    return data['current']['temperature_2m']

tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for provided coordinates in celsius.",
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {"type": "number"},
            "longitude": {"type": "number"}
        },
        "required": ["latitude", "longitude"],
        "additionalProperties": False
    },
    "strict": True
}]

input_messages = [{"role": "user", "content": "What's the weather like in West Lafayette, IN today?"}]

response = client.responses.create(
    model="gpt-4.1",
    input=input_messages,
    tools=tools,
)
# print(response.output_text)

tool_call = response.output[0]
args = json.loads(tool_call.arguments)

result = get_weather(args["latitude"], args["longitude"])

input_messages.append(tool_call)  # append model's function call message
input_messages.append({                               # append result message
    "type": "function_call_output",
    "call_id": tool_call.call_id,
    "output": str(result)
})

response_2 = client.responses.create(
    model="gpt-4.1",
    input=input_messages,
    tools=tools,
)
print(response_2.output_text)
raise ValueError

# %% Basic talk
from openai import OpenAI
import os
os.environ["OPENAI_API_KEY"] = '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'
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
import json

client = OpenAI()

prompt = """
Tell be briefly about the history of Indiana Pacers.
"""

# response = client.responses.create(
#     model="o4-mini",
#     reasoning={"effort": "low"},
#     input=[
#         {
#             "role": "user",
#             "content": prompt
#         }
#     ],
#     stream = True
# )

response = client.responses.create(
    model="o4-mini",
    reasoning={"effort": "low"},
    input=[
        {
            "role": "user",
            "content": prompt
        }
    ],
)
# for event in response:
#     print(event.response.output_text)
print(response.output_text)