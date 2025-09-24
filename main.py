import uvicorn
from src.api import app

if __name__ == "__main__":
    # Run the FastAPI application
    # The host '0.0.0.0' makes the server accessible from any IP address,
    # which is important for Docker deployment.
    uvicorn.run(app, host="0.0.0.0", port=8000)
