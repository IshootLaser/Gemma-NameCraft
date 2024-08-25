from fastapi import APIRouter
from utils.check_services import check
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get('health')
def health():
    service_ready = check()
    return JSONResponse(content=service_ready)
