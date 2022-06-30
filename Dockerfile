FROM nvcr.io/nvidia/cuda:11.7.0-base-ubuntu20.04
RUN apt update && DEBIAN_FRONTEND=noninteractive apt upgrade -y && DEBIAN_FRONTEND=noninteractive apt install -y git python3 python3-pip
COPY requirements.txt /workdir/requirements.txt
WORKDIR /workdir
RUN python3 -m pip install -r requirements.txt
COPY counterfit /workdir/counterfit
COPY examples/scripting/cf.py /workdir
RUN python3 -m counterfit

ENTRYPOINT ["python3","-m"]
CMD ["counterfit"]
