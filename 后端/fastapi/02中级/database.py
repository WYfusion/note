from fastapi import FastAPI, Depends
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from datetime import datetime

DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ConversationDB(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    message = Column(Text)
    response = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


class ConversationCreate(BaseModel):
    user_id: str
    message: str
    response: str


class ConversationResponse(BaseModel):
    id: int
    user_id: str
    message: str
    response: str
    created_at: datetime

    class Config:
        from_attributes = True


app = FastAPI()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/conversations/", response_model=ConversationResponse)
def create_conversation(conv: ConversationCreate, db: Session = Depends(get_db)):
    db_conv = ConversationDB(**conv.dict())
    db.add(db_conv)
    db.commit()
    db.refresh(db_conv)
    return db_conv


@app.get("/conversations/", response_model=list[ConversationResponse])
def list_conversations(user_id: str, db: Session = Depends(get_db)):
    return db.query(ConversationDB).filter(ConversationDB.user_id == user_id).all()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
