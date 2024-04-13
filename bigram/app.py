import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Params(BaseModel):
    text: str
@app.post("/bigram")
def bigram_endpoint(request: Params):
    return request.text


if __name__ == '__main__':
    uvicorn.run("bigram.app:app",port=8199,host='0.0.0.0',reload=True)

# if __name__ == '__main__':
#     model = BigramLanguageModel(vocab_size)
#     m = model.to(device)
#     idx = torch.tensor([1,2])
#     print("################################")
#     print(f"{idx=}")
#     print("#################################")
    # m.forward()










