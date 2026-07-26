from fastapi import FastAPI

app = FastAPI(
    title="AI Paper Reader API",
    version="0.1.0",
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "AI Paper Reader API is running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}