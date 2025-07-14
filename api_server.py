# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# ... (rest of the license header) ...

"""
A model worker executes the model.
"""
import argparse
import asyncio
import base64
import logging
import logging.handlers
import os
import sys
import tempfile
import threading
import shutil
import traceback
import uuid
from io import BytesIO

import torch
import trimesh
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from hy3dgen.shapegen import MeshSimplifier

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer, \
    MeshSimplifier
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.text2image import HunyuanDiTPipeline

LOGDIR = '.'

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None

def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


SAVE_DIR = 'gradio_cache'
os.makedirs(SAVE_DIR, exist_ok=True)

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("controller", f"{SAVE_DIR}/controller.log")


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


class ModelWorker:
    def __init__(self,
                 model_path='tencent/Hunyuan3D-2',
                 tex_model_path='tencent/Hunyuan3D-2',
                 subfolder='hunyuan3d-dit-v2-0-turbo',
                 device='cuda',
                 enable_tex=False):
        self.model_path = model_path
        self.worker_id = worker_id
        self.device = device
        logger.info(f"Loading the model {model_path} on worker {worker_id} ...")

        self.rembg = BackgroundRemover()
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            use_safetensors=True,
            torch_dtype=torch.float16,
            variant="fp16",
            device=device,
        )
        self.pipeline.enable_flashvdm(mc_algo='mc', topk_mode='merge')
        if enable_tex:
            self.pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(tex_model_path)

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate(self, uid, params):
        # ... (the multi-view handling part remains the same) ...
        multi_view_keys = ['front_image', 'back_image', 'left_image', 'right_image',
                           'top_image', 'bottom_image']
        has_multi_view = any(key in params for key in multi_view_keys)
        processed_images = {}
        if has_multi_view:
            logger.info("Multi-view generation detected.")
            for key in multi_view_keys:
                if key in params:
                    img_b64 = params[key]
                    img_pil = load_image_from_base64(img_b64)
                    img_rembg = self.rembg(img_pil)
                    processed_images[key] = img_rembg
            if 'front_image' in processed_images:
                params['image'] = processed_images['front_image']
            elif processed_images:
                params['image'] = next(iter(processed_images.values()))
            else:
                 raise ValueError("Multi-view generation requested but no valid images found.")
            params.update(processed_images)
        elif 'image' in params:
            logger.info("Single-view generation detected.")
            image = load_image_from_base64(params["image"])
            image = self.rembg(image)
            params['image'] = image
        else:
            raise ValueError("No input image(s) or text provided")

        if 'mesh' in params and 'glb' in params['mesh']:
            mesh = trimesh.load(BytesIO(base64.b64decode(params["mesh"])), file_type='glb')
        else:
            seed = params.get("seed", 1234)
            generation_params = params.copy()
            for key in list(generation_params.keys()):
                if key not in multi_view_keys and key not in ['image', 'generator', 'octree_resolution', 'num_inference_steps', 'guidance_scale', 'mc_algo']:
                    del generation_params[key]
            generation_params['generator'] = torch.Generator(self.device).manual_seed(seed)
            generation_params['octree_resolution'] = params.get("octree_resolution", 128)
            generation_params['num_inference_steps'] = params.get("num_inference_steps", 5)
            generation_params['guidance_scale'] = params.get('guidance_scale', 5.0)
            generation_params['mc_algo'] = 'mc'
            import time
            start_time = time.time()
            mesh = self.pipeline(**generation_params)[0]
            logger.info("--- Mesh generation took %s seconds ---" % (time.time() - start_time))

        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh, max_facenum=params.get('face_count', 10000))
        if params.get('texture', False):
            mesh = self.pipeline_tex(mesh, image)

        type = params.get('type', 'obj')
        with tempfile.NamedTemporaryFile(suffix=f'.{type}', delete=False) as temp_file:
            mesh.export(temp_file.name)
            save_path = os.path.join(SAVE_DIR, f'{str(uid)}.{type}')
            
            # Use shutil.move to handle moving files across different filesystems
            shutil.move(temp_file.name, save_path)

        torch.cuda.empty_cache()
        
        # Calculate origin
        origin = mesh.bounds.mean(axis=0).tolist()

        # Return path and origin
        return save_path, uid, origin

# ... (rest of the FastAPI app setup remains the same) ...
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],

    allow_headers=["*"],
)


@app.post("/generate")
async def generate(request: Request):
    logger.info("Worker generating...")
    params = await request.json()
    
    uid = uuid.uuid4()
    try:
        file_path, uid, origin = worker.generate(uid, params)
        
        # Encode file in base64
        with open(file_path, "rb") as f:
            encoded_mesh = base64.b64encode(f.read()).decode('utf-8')
        
        # Clean up the temporary file
        os.remove(file_path)

        # Return the generated file and origin as JSON
        return JSONResponse({
            "mesh_base64": encoded_mesh,
            "origin": origin,
            "filename": os.path.basename(file_path)
        })
    except ValueError as e:
        traceback.print_exc()
        print("Caught ValueError:", e)
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=400) # Use 400 for bad request
    except torch.cuda.CudaError as e:
        print("Caught torch.cuda.CudaError:", e)
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=500) # Use 500 for server error
    except Exception as e:
        print("Caught Unknown Error", e)
        traceback.print_exc()
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=500)


# The /send and /status endpoints can remain if you want async behavior,
# but for benchmarking, the synchronous /generate is better.
# ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument("--tex_model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-0-turbo')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument('--enable_tex', action='store_true')
    args = parser.parse_args()
    logger.info(f"args: {args}")

    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)

    worker = ModelWorker(model_path=args.model_path,
                         tex_model_path=args.tex_model_path,
                         subfolder=args.subfolder,
                         device=args.device,
                         enable_tex=args.enable_tex)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
