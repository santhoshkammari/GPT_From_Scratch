import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from trash_code.bigram.bigram import main

app = FastAPI()
class Params(BaseModel):
    text: str


@app.post("/bigram")
def bigram_endpoint(request: Params):
    generated_text = main(request.text)

    # for ch in generated_text:
    #     print(ch, end="", flush=True)
    #     time.sleep(0.05)

    return generated_text


if __name__ == '__main__':
    uvicorn.run("bigram.apis.bigram_api:app",port=8199,host='0.0.0.0',reload=True)











