FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Setting environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV HDF5_USE_FILE_LOCKING=FALSE
ENV NUMBA_CACHE_DIR=/tmp


# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
#RUN apt-get update && apt-get install -y --no-install-recommends \
#  build-essential libgl1-mesa-glx libglib2.0-0 libgeos-dev libvips-tools \
#  curl sudo htop git wget vim ca-certificates python3-openslide \
#  && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

COPY dUnet.py /app/
COPY multichannel.py /app/
COPY main.py /app/

CMD ["python", "main.py"]
