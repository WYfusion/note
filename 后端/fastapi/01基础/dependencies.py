from fastapi import FastAPI, Depends
from typing import Optional
from contextlib import asynccontextmanager

app = FastAPI()


class Database:
    def __init__(self):
        self.data = []
    
    def add(self, item: dict):
        self.data.append(item)
    
    def get_all(self):
        return self.data


_db_instance: Optional[Database] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db_instance
    _db_instance = Database()
    yield
    _db_instance = None


def get_db() -> Database:
    return _db_instance


async def get_current_user(token: str = None):
    if token == "valid":
        return {"username": "demo_user"}
    return None


@app.get("/")
def read_root():
    return {"message": "Dependency Injection Demo"}


@app.get("/db/items/")
def list_items(db: Database = Depends(get_db)):
    return db.get_all()


@app.post("/db/items/")
def add_item(item: dict, db: Database = Depends(get_db)):
    db.add(item)
    return {"status": "added", "items": db.get_all()}


@app.get("/protected/")
def protected_route(user: dict = Depends(get_current_user)):
    if not user:
        return {"error": "Unauthorized"}
    return {"user": user}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
