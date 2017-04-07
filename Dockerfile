FROM tensorflow/tensorflow:1.0.0

# Install librdkafka
#RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install
#      openssl tar && \
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install libpcre3 libssl-dev libpcre3-dev wget zip gcc 
RUN BUILD_DIR="$(mktemp -d)" && \
\
    export LIBRDKAFKA_VER=0.9.3 && wget -O "$BUILD_DIR/librdkafka.tar.gz" "https://github.com/edenhill/librdkafka/archive/v$LIBRDKAFKA_VER.tar.gz" && \
    mkdir -p $BUILD_DIR/librdkafka-$LIBRDKAFKA_VER && \
    tar \
      --extract \
      --file "$BUILD_DIR/librdkafka.tar.gz" \
      --directory "$BUILD_DIR/librdkafka-$LIBRDKAFKA_VER" \
      --strip-components 1 && \
\
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install g++ libssl-dev make musl-dev zlib1g-dev pax-utils &&  \
\
    cd "$BUILD_DIR/librdkafka-$LIBRDKAFKA_VER" && \
    ./configure \
      --prefix=/usr && \
    make -j "8" && \
    make install && \
\
    cd / && \
    rm -rf $BUILD_DIR

ADD requirements.txt /
ADD setup.py /
ADD facenet.py /
ADD serve.py /
ADD deps/docker-deps /confluent-kafka-python
COPY align /align

WORKDIR /confluent-kafka-python
RUN python setup.py install
WORKDIR /
RUN pip install -r /requirements.txt

CMD [ "mkdir", "/models"]
RUN mkdir /models
RUN wget https://storage.googleapis.com/youfie-983ce.appspot.com/20170216-091149.zip
RUN unzip 20170216-091149.zip
RUN mv 20170216-091149 /models
ARG LIBRDKAFKA_NAME="librdkafka"
ARG LIBRDKAFKA_VER="0.9.4"


CMD [ "python", "setup.py install"]
CMD [ "python", "/serve.py"]
