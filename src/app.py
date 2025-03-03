from fastapi import FastAPI
from routes.rag import rag_router

app = FastAPI()

app.include_router(router=rag_router)

if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="127.0.0.1", port=8000)