from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request, APIRouter

router = APIRouter(prefix="/prototype", tags=["prototype"])

@router.get("/song-writing")
def test():
    return "hello world"