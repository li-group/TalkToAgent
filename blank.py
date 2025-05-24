from openai import OpenAI

import os
os.environ["OPENAI_API_KEY"] = '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'
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

client = OpenAI()

prompt = """
Tell be briefly about the history of Indiana Pacers.
"""

response = client.responses.create(
    model="o4-mini",
    reasoning={"effort": "low"},
    input=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    stream = True
)

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