FROM  nvidia/cuda:11.1-devel-ubuntu20.04
LABEL maintainer="Azhar"
LABEL repository="textssl"

ENV DEBIAN_FRONTEND noninteractive
COPY ./cuda-keyring_1.0-1_all.deb cuda-keyring_1.0-1_all.deb
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip \
                   tesseract-ocr && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    mkl \
    gradio

RUN python3 -m pip install --no-cache-dir automodinit==0.16\
                        bidict==0.21.4\
                        GitPython==3.1.24\
                        lxml==4.7.1\
                        multiprocess==0.70.12.2\
                        numpy==1.21.4\
                        opencv_python_headless==4.5.5.62\
                        pandas==1.3.5\
                        pdf2image==1.16.0\
                        Pillow==9.0.0\
                        python_Levenshtein==0.12.2\
                        PyYAML==6.0\
                        tqdm==4.62.3
RUN apt-get update && apt-get install -y zip ipython3 poppler-utils nano
RUN python3 -m pip install --no-cache-dir --find-links https://download.pytorch.org/whl/torch_stable.html \
                            torch==1.9.0+cu111\
                            torchvision>=0.9.1+cu111
RUN python3 -m pip install --no-cache-dir gensim==4.2.0\
                        networkx==2.5 \
                        nltk==3.6.1 \
                        numpy==1.20.1 \
                        pandas==1.2.4 \
                        scikit_learn==1.1.1 \
                        scipy==1.6.2\
                        autopep8 joblib
RUN export TORCH=1.9.0 && \
    export CUDA=cu111 && \
    pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html && \
    pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html && \
    pip install torch-geometric

COPY ./inputrc /etc/inputrc                           
