from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()


class CustomException(Exception):
    def __init__(self, name: str):
        self.name = name


@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException):
    return JSONResponse(
        status_code=418,
        content={"detail": f"Custom error: {exc.name}"}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


items = {
    1: {"name": "Item 1", "price": 100},
    2: {"name": "Item 2", "price": 200},
}


@app.get("/items/{item_id}")
def read_item(item_id: int):
    if item_id not in items:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item {item_id} not found"
        )
    return items[item_id]


@app.post("/custom-error/")
def trigger_custom_error(name: str = "default"):
    if name == "error":
        raise CustomException(name)
    return {"message": "OK"}


@app.get("/validation-error/")
def validation_error():
    raise HTTPException(
        status_code=400,
        detail="Invalid input parameters"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
