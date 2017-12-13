FROM ubuntu

USER root

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

#https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
#wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.1.11-Linux-x86_64.sh -O ~/miniconda.sh && \

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENV PATH /opt/conda/bin:$PATH
# Configure environment
# ENV CONDA_DIR=/opt/conda \
#     SHELL=/bin/bash \
#     NB_USER=jovyan \
#     NB_UID=1000 \
#     NB_GID=100

# RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
#     mkdir -p $CONDA_DIR && \
#     chown $NB_USER:$NB_GID $CONDA_DIR

RUN apt-get update
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y git wget
RUN apt-get install bzip2
RUN apt-get install -y gcc
RUN apt-get install -y g++
RUN apt-get install -y libgtk2.0-0
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get install libglib2.0-0

RUN apt-get install libc6-i386
RUN apt-get install -y libsm6 libxrender1

RUN conda config --set always_yes yes
RUN conda update --yes conda
RUN conda info -a
RUN CONDA_SSL_VERIFY=false conda update pyopenssl

ADD . /CaImAn/
WORKDIR /CaImAn/

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# RUN conda install python=3.6
# RUN conda create -n root python=3.6
RUN conda env update python=3.6 -f environment.yml -n caiman 
RUN source activate caiman && \
    conda install -c anaconda pyqt && \
    #conda install anaconda-nb-extensions -c anaconda-nb-extensions && \
    jupyter nbextension enable --py --sys-prefix widgetsnbextension && \
    python setup.py install && python setup.py build_ext -i

RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

EXPOSE 8888
# CMD "source activate caiman && jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root"
# CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]