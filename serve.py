import json, argparse, time
import os, os.path
import uuid
import base64
from PIL import Image
from io import BytesIO
from pprint import pprint

from scipy import misc
import tensorflow as tf
import numpy as np
import requests
from pykafka import KafkaClient

import facenet
import align.detect_face as detect_face


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


def begin_message_consumption(consumer):
    photos = []
    while 1:
        msg = consumer.consume()
        if msg:
            image = json.loads(msg.value)
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
                    image['faces'][ndx].update({
                        'bb': face['bb'].tolist(),
                        'embedding': embs[ndx].tolist()
                    })
                pprint(image)
                print '\n' * 8


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="models/20170216-091149",
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

    print("Loading the model...")
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
                    print("starting the consumer")
                    client = KafkaClient("104.196.19.209:9092")
                    topic = client.topics["facenet"]
                    consumer = topic.get_balanced_consumer(
                        consumer_group="charlie",
                        auto_commit_enable=True,
                        zookeeper_connect='104.196.19.209:2181')
                    begin_message_consumption(consumer)
