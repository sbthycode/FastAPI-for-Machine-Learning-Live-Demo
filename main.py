import io

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from enum import Enum
from ml import obtain_image, clf, iris

app = FastAPI()
#### Inital exploration of the FastAPI framework ####

@app.get("/")
async def root():
    return {"message": "Hello World"}

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

# We can use our own class using enum to which we want the input to belong to:
@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}

# We can use the same class to validate the input:
@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    return {"file_path": file_path}


# We can extract and return data from a database:
fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]
@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]

### Using FastAPI for Machine Learning ###
# We can use FastAPI to create a machine learning API

# This is how we validatae the input data
class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float

# We predict(through the post method) the class of the flower based on the input data and return the class
@app.post("/predict")
def predict(data : request_body):
    test_data = [[
            data.sepal_length, 
            data.sepal_width, 
            data.petal_length, 
            data.petal_width
    ]]
    class_idx = clf.predict(test_data)[0]
    return { 'class' : iris.target_names[class_idx]}

### Using FastAPI for Image Generation using stable diffusion ###

@app.get("/generate")
def generate_image(
    prompt: str,
    *,
    seed: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
):
    image = obtain_image(
        prompt,
        num_inference_steps=num_inference_steps,
        seed=seed,
        guidance_scale=guidance_scale,
    )
    image.save("image.png")
    return FileResponse("image.png")


@app.get("/generate-memory")
def generate_image_memory(
    prompt: str,
    *,
    seed: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
):
    image = obtain_image(
        prompt,
        num_inference_steps=num_inference_steps,
        seed=seed,
        guidance_scale=guidance_scale,
    )
    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/png")


    
