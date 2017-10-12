FROM nvidia/cuda:8.0-cudnn5-devel

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    apt-get update && \
    apt-get install -y wget git libhdf5-dev g++ graphviz ucto && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-3.9.1-Linux-x86_64.sh && \
    echo "6c6b44acdd0bc4229377ee10d52c8ac6160c336d9cdd669db7371aa9344e1ac3 *Miniconda3-3.9.1-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-3.9.1-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-3.9.1-Linux-x86_64.sh

ENV NB_USER keras
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown keras $CONDA_DIR -R && \
    mkdir -p /src && \
    chown keras /src

USER keras

# Python
ARG python_version=3.5.2
ARG tensorflow_version=1.0.1-cp35-cp35m

RUN conda install -y python=${python_version} && \
    pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-${tensorflow_version}-linux_x86_64.whl && \
    pip install git+git://github.com/Theano/Theano.git && \
    pip install ipdb pytest pytest-cov python-coveralls coverage==3.7.1 pytest-xdist pep8 pytest-pep8 pydot_ng && \
    conda install Pillow scikit-learn jupyter pandas matplotlib nose pyyaml six h5py nltk cython && \
    pip install git+git://github.com/fchollet/keras.git && \
    conda clean -yt && \
    python -m nltk.downloader punkt

ADD theanorc /home/keras/.theanorc

# Set up notebook config
COPY jupyter_notebook_config.py /home/keras/.jupyter/

ENV PYTHONPATH='/src/:$PYTHONPATH'

WORKDIR /src

EXPOSE 8888
EXPOSE 6006

CMD /bin/bash -c 'jupyter notebook --no-browser --ip=* "$@"'



