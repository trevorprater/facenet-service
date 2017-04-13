import json, argparse, time
import os, os.path
import sys
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
from psycopg2.extras import execute_values
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
    'bootstrap.servers': '10.142.0.3:9092',
    'group.id': 'bravo',
    'session.timeout.ms': 10000,
    'api.version.request': True,
    'receive.message.max.bytes': 204859424,
    'default.topic.config': {
        'auto.offset.reset': 'smallest'
    }
}


def create_new_consumer():
    consumer = Consumer(**KAFKA_CONF)
    consumer.subscribe(['bluefin'])
    return consumer


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


def insert_photos_to_db(photos):
    logging.exception(
        "{}: entered insert_photos_to_db function".format(time.time()))
    faces = []
    for photo in photos:
        photo['id'] = str(uuid.uuid4())
        for face in photo['faces']:
            faces.append((
                photo['id'], face['bb']['top_left_x'],
                face['bb']['top_left_y'], face['bb']['bottom_right_x'],
                face['bb']['bottom_right_y'],
                '{' + ",".join([str(emb) for emb in face['embedding']]) + '}'))
    logging.exception(
        "{}: created data structures for db insertion".format(time.time()))

    logging.exception("{}: begin insert photos to db".format(time.time()))
    execute_values(
        CUR,
        "INSERT INTO photos(id, url, parent_url, sha256) VALUES %s ON CONFLICT(url, sha256) DO NOTHING",
        [(p['id'], p['url'], p['parent_url'], p['sha256']) for p in photos])
    logging.exception("{}: end insert photos to db".format(time.time()))

    logging.exception("{}: begin insert faces to db".format(time.time()))
    execute_values(
        CUR,
        "INSERT INTO faces(photo_id, top_left_x, top_left_y, bottom_right_x, bottom_right_y, feature_vector) SELECT val.photo_id, val.top_left_x, val.top_left_y, val.bottom_right_x, val.bottom_right_y, val.feature_vector FROM ( VALUES %s ) val (photo_id, top_left_x, top_left_y, bottom_right_x, bottom_right_y, feature_vector) JOIN photos USING (id) ON CONFLICT(photo_id, top_left_x, top_left_y, bottom_right_x, bottom_right_y) DO NOTHING",
        faces)
    logging.exception("{}: end insert faces to db".format(time.time()))

    logging.exception("{}: begin conn.commit()".format(time.time()))
    CONN.commit()
    logging.exception("{}: end conn.commit()".format(time.time()))


def begin_message_consumption(consumer):
    images = []
    while 1:
        logging.exception("{}: begin mesage consume".format(time.time()))
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue

        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                logging.exception("%% %s [%d] reached end at offset %d\n" %
                                  (msg.topic(), msg.partition(), msg.offset()))
            else:
                logging.exception(msg.error())

        if msg and not msg.error():
            logging.exception("{}: message received".format(time.time()))
            image = json.loads(msg.value())
            logging.exception(
                "{}: begin load and align data".format(time.time()))
            image = load_and_align_data(image, args.image_size, args.margin,
                                        args.gpu_memory)
            logging.exception(
                "{}: end load and align data".format(time.time()))
            if len(image['faces']) > 0:
                face_bytes = np.stack(
                    [face['prewhitened'] for face in image['faces']])
                logging.exception(
                    "{}: begin embedding session".format(time.time()))
                embs = embed_sess.run(
                    embeddings,
                    feed_dict={
                        images_placeholder: face_bytes,
                        phase_train_placeholder: False
                    })
                logging.exception(
                    "{}: end embedding session".format(time.time()))
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
                logging.exception('{}: processed {} faces'.format(
                    time.time(), len(image['faces'])))
                images.append(image)
                if len(images) >= 20:
                    logging.exception(
                        "{}: calling insert_photos_to_db".format(time.time()))
                    insert_photos_to_db(images)
                    logging.exception("{}: return from insert_photos_to_db".
                                      format(time.time()))
                    images = []


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

    logging.exception("{}: loading the model...".format(time.time()))
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
                    logging.exception(
                        "{}: starting the consumer".format(time.time()))
                    consumer = create_new_consumer()
                    begin_message_consumption(consumer)
