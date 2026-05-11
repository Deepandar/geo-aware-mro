import time

from starlette.middleware.base import BaseHTTPMiddleware


class RequestLoggingMiddleware(BaseHTTPMiddleware):

    async def dispatch(
        self,
        request,
        call_next,
    ):

        start = time.time()

        response = await call_next(request)

        duration = round(
            time.time() - start,
            4,
        )

        print(
            f"[REQUEST] "
            f"{request.method} "
            f"{request.url.path} "
            f"{response.status_code} "
            f"{duration}s"
        )

        return response
