import os
import time
import base64
import json
import asyncio
import httpx
import boto3
from botocore.client import Config
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize R2 client at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global s3_client, http_client
    s3_client = boto3.client(
        's3',
        endpoint_url=os.environ['R2_ENDPOINT_URL'],
        aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
        config=Config(signature_version='s3v4')
    )
    http_client = httpx.AsyncClient()
    yield
    await http_client.aclose()

app.router.lifespan_context = lifespan

async def generate_image(prompt: str):
    try:
        # Verify FLUX_API_KEY is set
        if "FLUX_API_KEY" not in os.environ:
            raise HTTPException(
                status_code=500,
                detail="FLUX_API_KEY environment variable not set"
            )

        headers = {
            'accept': 'application/json',
            'authorization': f'Bearer {os.environ["FLUX_API_KEY"]}',
        }

        files = {
            'prompt': (None, prompt),
            'model': (None, 'flux_1_schnell'),
            'size': (None, '16_9'),
            'lora': (None, ''),
            'style': (None, 'no_style'),
        }

        response = await http_client.post(
            'https://api.freeflux.ai/v1/images/generate',
            headers=headers,
            files=files,
            timeout=30.0
        )
        response.raise_for_status()

        response_json = response.json()
        image_data_url = response_json.get('result')
        
        if not image_data_url or not image_data_url.startswith("data:image/png;base64,"):
            raise HTTPException(
                status_code=500,
                detail="Invalid image data response"
            )

        return image_data_url.split(",")[1]
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Image generation failed: {str(e)}"
        )
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Invalid API response format"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image generation error: {str(e)}"
        )

async def upload_to_r2(image_data: str) -> str:
    try:
        # Verify required R2 environment variables are set
        required_vars = ['R2_BUCKET_NAME', 'R2_PUBLIC_DOMAIN']
        for var in required_vars:
            if var not in os.environ:
                raise HTTPException(
                    status_code=500,
                    detail=f"{var} environment variable not set"
                )

        image_bytes = base64.b64decode(image_data)
        timestamp = int(time.time())
        object_name = f"generated_image_{timestamp}.png"
        
        await asyncio.to_thread(
            s3_client.put_object,
            Bucket=os.environ['R2_BUCKET_NAME'],
            Key=object_name,
            Body=image_bytes,
            ContentType='image/png'
        )
        
        return f"https://{os.environ['R2_PUBLIC_DOMAIN']}/{object_name}"
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"R2 upload failed: {str(e)}"
        )

@app.post("/generate")
async def generate_endpoint(
    payload: dict,
    background_tasks: BackgroundTasks
):
    try:
        prompt = payload.get("prompt")
        if not prompt:
            raise HTTPException(
                status_code=400,
                detail="Prompt is required"
            )

        base64_image = await generate_image(prompt)
        image_url = await upload_to_r2(base64_image)
        
        return {"image_url": image_url}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
