#sudo /usr/local/python310/bin/python3 -m pip install onnxruntime
#sudo /usr/local/python310/bin/python3 -m pip install opencv-python
#sudo /usr/local/python310/bin/python3 -m pip install torchaudio-2.0.2+rocm5.4.2-cp310-cp310-linux_x86_64.whl
#sudo /usr/local/python310/bin/python3 -m pip install torchvision-0.15.2+rocm5.4.2-cp310-cp310-linux_x86_64.whl
#sudo /usr/local/python310/bin/python3 -m pip install matplotlib
#sudo /usr/local/python310/bin/python3 -m pip install gradio
#sudo /usr/local/python310/bin/python3 -m pip install transformers
#sudo /usr/local/python310/bin/python3 -m pip install datasets
#sudo apt-get install -y liblzma-dev
#安装 backports.lzma-0.0.14.tar.gz  参考 F:\cmdmax\reference\Python\相关资料\Python 3.10 安装lzma
#sudo /usr/local/python310/bin/python3 -m pip install torchkeras
#sudo /usr/local/python310/bin/python3 -m pip install peft
#sudo /usr/local/python310/bin/python3 -m pip install Namespace
#sudo /usr/local/python310/bin/python3 -m pip install sentencepiece
#sudo /usr/local/python310/bin/python3 -m pip install jupyter
#sudo /usr/local/python310/bin/python3 -m pip install jupyterlab
#sudo /usr/local/python310/bin/python3 -m pip install ipython

#python.exe -m pip install --upgrade pip
#cd F:\software\"NVIDIA AI"\"PyTorch 2.0.1"
#F:
#python -m pip install torchaudio-2.0.2+cu118-cp310-cp310-win_amd64.whl
#python -m pip install matplotlib
#python -m pip install pandas
#python -m pip install transformers
#python -m pip install datasets
#python -m pip install torchkeras
#python -m pip install peft
#python -m pip install accelerate
#python -m pip install gradio
#python -m pip install fastapi
#python -m pip install numpy
#python -m pip install opencv-python
#python -m pip install onnxruntime
#python -m pip install onnx

import sys
import os
import sys
import platform
import subprocess

print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -sys");
print(sys.version)
print(os.path.abspath(sys.argv[0]))
print(sys.executable)

print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -PyTorch");
try:
    import torch
    print("torch",torch.__version__)    #Pytorch版本
    print("torch.version.cuda",torch.version.cuda)   #CUDA版本
    print("cuda.is_available",torch.cuda.is_available()) #CUDA检测是否可用
    print("cudnn.version",torch.backends.cudnn.version())   #CUDNN版本
except Exception as e:
    print(str(e))

print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -Cython");
try:
    import Cython
    print("Cython",Cython.__version__);
except Exception as e:
    print(str(e))

try:
    import torchaudio
    print("torchaudio",torchaudio.__version__) 
except Exception as e:
    print(str(e))

try:
    import IPython.display as ipd
except Exception as e:
    print(str(e))

try:
    import matplotlib.pyplot as plt
except Exception as e:
    print(str(e))

try:
    import pandas as pd 
except Exception as e:
    print(str(e))

try:
    import transformers
except Exception as e:
    print(str(e))

try:
    import datasets 
except Exception as e:
    print(str(e))

try:
    from torch.utils.data import Dataset,DataLoader 
except Exception as e:
    print(str(e))

try:
    from transformers import AutoModel,AutoTokenizer,AutoConfig,DataCollatorForSeq2Seq
except Exception as e:
    print(str(e))

try:
    from torchkeras.chat import ChatGLM
except Exception as e:
    print(str(e))

try:
    from argparse import Namespace
except Exception as e:
    print(str(e))

try:
    from peft import get_peft_model, AdaLoraConfig, TaskType
except Exception as e:
    print(str(e))

try:
    from peft import PeftModel
except Exception as e:
    print(str(e))

try:
    from torchkeras import KerasModel 
except Exception as e:
    print(str(e))

try:
    from accelerate import Accelerator 
except Exception as e:
    print(str(e))

def get_gpu_vendor():
    if platform.system() == "Windows":

        try:
            output = subprocess.check_output(["nvidia-smi"]).decode(errors="ignore")
            if "NVIDIA" in output:
                return "NVIDIA"
        except subprocess.CalledProcessError as e:
            print(str(e))
        try:
            output = subprocess.check_output(["amdgpu-pro-px"])
            if "AMD" in output.decode():
                return "AMD"
        except subprocess.CalledProcessError as e:
            print(str(e))

        return "Unknown"
    elif platform.system() == "Linux":
        try:
            output = subprocess.check_output(["lspci", "-v"]).decode(errors="ignore")
            if "NVIDIA" in output:
                return "NVIDIA"
            elif "AMD" in output:
                return "AMD"
        except subprocess.CalledProcessError as e:
            print(str(e))

        return "Unknown"
    else:
        return "Unsupported"

gpu="Unknown"
gpu=get_gpu_vendor();
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -gradio");
try:
    import gradio
    print("gradio",gradio.__version__)
except Exception as e:
    print(str(e))
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -fastapi");
try:
    import fastapi
    print("fastapi",fastapi.__version__)
except Exception as e:
    print(str(e))
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -numpy");
try:
    import numpy
    print("numpy",numpy.__version__)
except Exception as e:
    print(str(e))
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -CV2");
try:
    import cv2
    print("opencv-python",cv2.__version__)
except Exception as e:
    print(str(e))
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ONNXRuntime");
try:
    import onnxruntime as ort
    print("onnxruntime",ort.__version__)
except Exception as e:
    print(str(e))
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -TensorRT");
try:
    if(gpu=="NVIDIA"):
        import tensorrt
        print("tensorrt",tensorrt.__version__)
    else:
        print("not support")  
except Exception as e:
    print(str(e))
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -ONNX");
try:
    import onnx
    print("onnx",onnx.__version__)
except Exception as e:
    print(str(e))
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -GPU");
print("gpu",gpu)
if platform.system().lower()=="windows":
    print("not support")
elif platform.system().lower()=="linux":
    if(gpu=="NVIDIA"):
        os.system("lspci | grep -i nvidia");
    else:
        os.system("lspci | grep -i amd");
        
if platform.system().lower()=="windows":
    print("windows");
elif platform.system().lower()=="linux":
    os.system("hostnamectl");
print(platform.system().lower());#系统平台
print(os.getcwd()); # 当前目录路径


print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -pip");
pip_cmd="pip --version";# windows
pip3_cmd="pip3 --version";
if platform.system().lower()=="linux":
    pip_cmd="/usr/local/python310/bin/pip --version";#linux
    pip3_cmd="/usr/local/python310/bin/pip3 --version";
fa=os.popen(pip_cmd);
print(fa.read());
fa.close();
fb=os.popen(pip3_cmd);
print(fb.read());
fb.close();
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -Python");
print("sys",sys.version);
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -CUDA or ROCm");
if platform.system().lower()=="windows":
    if(gpu=="NVIDIA"):
        f=os.popen("nvcc -V");
        print(f.read());
        f.close();
    else:
        print("not support")
elif platform.system().lower()=="linux":
    if(gpu=="NVIDIA"):
        os.system("nvcc -V");
    else:
        os.system("dpkg -l|grep rocm-dev");
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -GPU Drivers");
if platform.system().lower()=="windows":
    if(gpu=="NVIDIA"):
        print("nvidia-smi");
        fc=os.popen("nvidia-smi");
        print(fc.read());
        fc.close();
    else:
        print("not support")
elif platform.system().lower()=="linux":
    if(gpu=="NVIDIA"):
        os.system("nvidia-smi");
    else:
        os.system("rocm-smi");
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -cuDNN");
print("version=",torch.backends.cudnn.version())
if platform.system().lower()=="windows":
    if(gpu=="NVIDIA"):
       #如果没有安装CUDA11.7
       try:
         target_directory = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\extras\demo_suite"
         os.chdir(target_directory)
       except Exception as e:
         print(str(e))
	 #那就去找CUDA11.8
         target_directory = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\demo_suite"
         os.chdir(target_directory)
       try:
         subprocess.run(["bandwidthTest.exe"], check=True)
       except subprocess.CalledProcessError:
         print("Error executing the .exe file.")
    else:
       print("not support")
elif platform.system().lower()=="linux":
    if(gpu=="NVIDIA"):
       #os.system("cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2");
       f=os.popen("cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2");
       print(f.read());
       f.close();
    else:
        print("not support")
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -jupyter");
jupyter_cmd="jupyter --version";# windows
if platform.system().lower()=="linux":
    jupyter_cmd="/usr/local/python310/bin/jupyter --version";#linux
fa=os.popen(jupyter_cmd);
print(fa.read());
fa.close();