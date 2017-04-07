import json, argparse, time
import os, os.path
import uuid
import base64
from PIL import Image
from io import BytesIO
import uuid
import logging

from scipy import misc
import tensorflow as tf
import numpy as np
import requests
import psycopg2
from confluent_kafka import Consumer, KafkaError

import facenet
import align.detect_face as detect_face

DB_HOST, DB_PORT = os.getenv("FACENET_DB_ADDR").split(":")
DB_PASSWORD = os.getenv("FACENET_DB_PASSWORD")
DB_USER = os.getenv("FACENET_DB_USER")

CONN = psycopg2.connect("dbname=facenet user={} host={} password={}".format(
    DB_USER, DB_HOST, DB_PASSWORD))
CUR = CONN.cursor()

KAFKA_CONF = {
    'bootstrap.servers': '104.196.19.209:9092',
    'group.id': 'charlie',
    'session.timeout.ms': 6000,
    'default.topic.config': {
        'auto.offset.reset': 'smallest'
    }
}


def create_new_consumer():
    failures = 0

    while failures <= 5:
        try:
            consumer = confluent_kafka.Consumer(**conf)
            c.subscribe(['facenet-test'])

            return consumer
        except Exception as e:
            failures += 1
            time.sleep(5)

        return None


def load_and_align_data(image, image_size, margin, gpu_memory_fraction):

    im = Image.open(BytesIO(base64.b64decode(image['b64_bytes'])))
    img = misc.fromimage(im, flatten=False, mode='RGB')
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet,
                                                threshold, factor)
    image['faces'] = []
    for box_ctr, bounding_box in enumerate(bounding_boxes):
        det = np.squeeze(bounding_boxes[box_ctr, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(
            cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        image['faces'].append({'prewhitened': prewhitened, 'bb': bb})
    return image


def insert_photo_to_db(photo):
    photo_id = str(uuid.uuid4())

    try:
        CUR.execute(
            "INSERT INTO photos(id, url, parent_url, sha256) VALUES ('{}', '{}', '{}', '{}')".
            format(photo_id, photo['url'], photo['parent_url'], photo[
                'sha256']))
        for face in photo['faces']:
            CUR.execute(
                "INSERT INTO faces(photo_id, top_left_x, top_left_y, bottom_right_x, bottom_right_y, feature_vector) VALUES ('{}', '{}', '{}', '{}', '{}', '{}')".
                format(photo_id, face['bb']['top_left_x'], face['bb'][
                    'top_left_y'], face['bb']['bottom_right_x'], face['bb'][
                        'bottom_right_y'], '{' + ",".join(
                            [str(emb) for emb in face['embedding']]) + '}'))
        CONN.commit()
    except psycopg2.IntegrityError as e:
        CONN.rollback()


def begin_message_consumption(consumer):
    num_failures = 0
    while 1:
        msg = None
        try:
            msg = c.poll(timeout=1.0)
        except Exception as e:
            time.sleep(5)
            num_failures += 1
            consumer = create_new_consumer()
            if num_failures > 10:
                raise Exception("CANNOT CONNECT TO KAFKA: {}".format(e))

        if msg is None:
            continue

        if msg.error():
            if msg.error().code() == KafkaError.__PARTITION_EOF:
                sys.stderr.write("%% %s [%d] reached end at offset %d\n" %
                                 (msg.topic(), msg.partition(), msg.offset()))
            elif msg.error():
                raise KafkaException(msg.error())
        if msg:
            num_failures = 0
            image = json.loads(msg.value())
            image = load_and_align_data(image, args.image_size, args.margin,
                                        args.gpu_memory)
            if len(image['faces']) > 0:
                face_bytes = np.stack(
                    [face['prewhitened'] for face in image['faces']])
                embs = embed_sess.run(
                    embeddings,
                    feed_dict={
                        images_placeholder: face_bytes,
                        phase_train_placeholder: False
                    })
                for ndx, face in enumerate(image['faces']):
                    bb_dict = dict(
                        zip([
                            'top_left_x', 'top_left_y', 'bottom_right_x',
                            'bottom_right_y'
                        ], face['bb'].tolist()))
                    image['faces'][ndx].update({
                        'bb': bb_dict,
                        'embedding': embs[ndx].tolist()
                    })
                logging.info(image['url'],
                             "num faces = {}".format(len(image['faces'])))
                insert_photo_to_db(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="/models/20170216-091149",
        type=str,
        help="directory containing the model")
    parser.add_argument(
        "--gpu_memory", default=1.0, type=float, help="GPU memory per process")
    parser.add_argument("--image_size", default=160, type=int)
    parser.add_argument("--margin", default=44, type=int)
    args = parser.parse_args()

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps' threshold
    factor = 0.709  # scale factor

    logging.info("Loading the model...")
    with tf.Graph().as_default() as graph:
        with tf.Session() as embed_sess:
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=args.gpu_memory)
            meta_file, ckpt_file = facenet.get_model_filenames(
                os.path.expanduser(args.model_dir))
            facenet.load_model(args.model_dir, meta_file, ckpt_file)
            images_placeholder = graph.get_tensor_by_name("input:0")
            embeddings = graph.get_tensor_by_name("embeddings:0")
            phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")

            with tf.Graph().as_default():
                detect_sess = tf.Session(config=tf.ConfigProto(
                    gpu_options=gpu_options, log_device_placement=False))
                with detect_sess.as_default():
                    pnet, rnet, onet = detect_face.create_mtcnn(detect_sess,
                                                                None)
                    logging.info("starting the consumer")
                    consumer = create_new_consumer()
                    begin_message_consumption(consumer)
