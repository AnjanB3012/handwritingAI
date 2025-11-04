from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from SystemManager import SystemManager
import uvicorn

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

system_manager = SystemManager("config.json", "input_data.json", "needed_data.json")

class StrokeDataRequest(BaseModel):
    stroke_data: dict

class SystemStateRequest(BaseModel):
    active_state: bool

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/get_next_needed_data")
def get_next_needed_data():
    data = system_manager.get_next_needed_data()
    if data is None:
        return {"message": "No more data to process"}
    return data

@app.post("/update_InData")
def update_InData(request: StrokeDataRequest):
    system_manager.update_InData(request.stroke_data)
    return {"message": "InData updated successfully"}

@app.post("/change_system_states")
def change_system_state(request: SystemStateRequest):
    system_manager.set_active_state(request.active_state)
    return {"message": "System state changed successfully"}

@app.get("/get_system_states")
def get_system_states():
    return {"active_state": system_manager.get_active_state()}

@app.get("/get_system_name")
def get_system_name():
    return {"name": system_manager.get_name()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)