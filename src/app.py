from fastapi import FastAPI
from routes.rag import rag_router
from routes.user_detail import user_router

app = FastAPI()

routers = [rag_router, user_router]
for route in routers:
   app.include_router(route)
# app.include_router(router=rag_router)

if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="127.0.0.1", port=8000) 