conda create -n Hunyuan3D python==3.10.9<br/>
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124
<br/>
conda activate Hunyuan3D<br/>
python3 gradio_app.py --model_path tencent/Hunyuan3D-2mv --subfolder hunyuan3d-dit-v2-mv-turbo --enable_flashvdm<br/>
python test.py <br/>
python mv-turbo-flash.py --input front.jpg back.jpg left.jpg --output output.glb<br/>

python api_server.py --host 0.0.0.0 --port 8080<br/>
python apitest.py <br/>
python apitest.py --with-texture