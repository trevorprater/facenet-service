FROM python:2.7

ADD requirements.txt /
ADD setup.py /
ADD facenet.py /
ADD serve.py /
COPY models /models
COPY align /align


RUN pip install -r requirements.txt

CMD [ "python", "setup.py install"]
CMD [ "python", "serve.py"]
