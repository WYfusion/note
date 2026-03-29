from fastapi import FastAPI, Response, status
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    description: Optional[str] = None


class User(BaseModel):
    username: str
    email: str
    items: Optional[List[Item]] = None


users_db = {}


@app.get("/users/{user_id}", response_model=User)
def get_user(user_id: int):
    return users_db.get(user_id)


@app.post("/users/", response_model=User, status_code=status.HTTP_201_CREATED)
def create_user(user: User):
    user_id = len(users_db) + 1
    users_db[user_id] = user
    return user


@app.put("/users/{user_id}", response_model=User)
def update_user(user_id: int, user: User):
    users_db[user_id] = user
    return user


@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, response: Response):
    if user_id in users_db:
        del users_db[user_id]
    return response


@app.get("/users/{user_id}/export")
def export_user(user_id: int):
    user = users_db.get(user_id)
    if not user:
        return {"error": "User not found"}
    return user


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
