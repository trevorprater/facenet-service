FROM tensorflow/tensorflow:1.0.0
#python:2.7

ADD requirements.txt /
ADD setup.py /
ADD facenet.py /
ADD serve.py /
#COPY models /models
COPY align /align


RUN pip install -r /requirements.txt

CMD [ "mkdir", "/models"]
RUN mkdir /models
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install libpcre3 libssl-dev libpcre3-dev wget zip gcc 
RUN wget https://storage.googleapis.com/youfie-983ce.appspot.com/20170216-091149.zip
RUN unzip 20170216-091149.zip
RUN mv 20170216-091149 /models

CMD [ "python", "setup.py install"]
CMD [ "python", "/serve.py"]

