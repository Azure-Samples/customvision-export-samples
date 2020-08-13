FROM ubuntu:18.04

RUN apt-get update && apt-get install -y curl gnupg software-properties-common
RUN curl -fsSL https://apt.repos.intel.com/openvino/2020/GPG-PUB-KEY-INTEL-OPENVINO-2020 | apt-key add -
RUN apt-add-repository "deb https://apt.repos.intel.com/openvino/2020 all main"
RUN apt-get update && apt-get install -y intel-openvino-dev-ubuntu18-2020.2.130 --no-install-recommends

RUN apt-get update && apt-get install -y python3.7 libpython3.7 python3-pip --no-install-recommends
RUN python3.7 -m pip install -U pip
RUN python3.7 -m pip install --no-cache-dir setuptools
RUN python3.7 -m pip install --no-cache-dir -r /opt/intel/openvino/deployment_tools/model_optimizer/requirements.txt

ENV PYTHONPATH=/opt/intel/openvino_2020.2.130/python/python3.7/:$PYTHONPATH
ENV LD_LIBRARY_PATH=/opt/intel/openvino_2020.2.130/deployment_tools/inference_engine/lib/intel64:/opt/intel/openvino_2020.2.130/deployment_tools/ngraph/lib:/opt/intel/openvino_2020.2.130/deployment_tools/inference_engine/external/tbb/lib:$LD_LIBRARY_PATH

RUN python3.7 -m pip install --no-cache-dir Pillow