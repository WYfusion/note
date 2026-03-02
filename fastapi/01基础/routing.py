from fastapi import FastAPI, Path, Query
from pydantic import BaseModel
from typing import Optional

app = FastAPI()


class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None


@app.get("/users/{user_id}")
def get_user(user_id: int = Path(..., gt=0)):
    return {"user_id": user_id}


@app.get("/search")
def search(
    q: str = Query("default", min_length=3),
    limit: int = Query(10, ge=0, le=100)
):
    return {"q": q, "limit": limit}


@app.post("/items/")
def create_item(item: Item):
    return item


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_id": item_id, **item.dict()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
