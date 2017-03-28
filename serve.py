import json, argparse, time
import os, os.path
import uuid

from scipy import misc
from flask import Flask, request
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import requests

import facenet
import align.detect_face as detect_face

app = Flask(__name__)
cors = CORS(app)

def _download_image(url):
    img_uuid = str(uuid.uuid4())
    filepath = "/tmp/" + img_uuid
    exec_cmd = "wget -O {} {}".format(filepath, url)
    os.system(exec_cmd)
    if os.path.isfile(filepath):
        return filepath

def load_and_align_data(images_metadata, image_size, margin, gpu_memory_fraction):
    faces = []
    for url in images_metadata.keys():
        img = misc.imread(images_metadata[url]['filepath'])
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ =  detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        for box_ctr, bounding_box in enumerate(bounding_boxes):
            det = np.squeeze(bounding_boxes[box_ctr,0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            faces.append({'url': url, 'prewhitened': prewhitened, 'bb': bb})
            #img_list.append((prewhitened, image_paths[i], bb))
            #images = np.stack(img_list)
    #return images
    #return img_list
    return faces

@app.route("/api/embed", methods=['POST'])
def embed():
    start = time.time()

    data = request.data.decode("utf-8")
    if data == "":
        params = request.form
        photo_urls = json.loads(params["photos"])
    else:
        params = json.loads(data)
        photo_urls  = params["photos"]

    images_metadata = {}
    for url in photo_urls:
        filepath = _download_image(url)
        if filepath:
            images_metadata[url] = {'filepath': filepath, 'faces': []}

    #images = load_and_align_data(filepaths, args.image_size, args.margin, args.gpu_memory)
    faces = load_and_align_data(images_metadata, args.image_size, args.margin, args.gpu_memory)
    face_bytes = [face['prewhitened'] for face in faces]
    images = np.stack(face_bytes)

    embs = embed_sess.run(embeddings, feed_dict={images_placeholder: images, phase_train_placeholder: False})

    print('number of faces to embed: {}'.format(len(faces)))
    print('number of embeddings returned: {}'.format(len(embs)))

    for i, face in enumerate(faces):
        url = face['url']
        bb = face['bb']
        images_metadata[url]['faces'].append({'bb': bb.tolist(), 'embedding': embs[i].tolist()})
        try:
            os.system('rm {} &'.format(images_metadata[url]['filepath']))
            del images_metadata[url]['filepath']
        except KeyError:
            pass

    json_data = json.dumps(images_metadata)
    return json_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/20170216-091149", type=str, help="directory containing the model")
    parser.add_argument("--gpu_memory", default=1.0, type=float, help="GPU memory per process")
    parser.add_argument("--image_size", default=160, type=int)
    parser.add_argument("--margin", default=44, type=int)
    args = parser.parse_args()

    minsize = 20 # minimum size of face
    threshold = [0.6, 0.7, 0.7] # three steps' threshold
    factor = 0.709 # scale factor

    print("Loading the model...")
    #graph = load_graph(args.frozen_model_filename)
    with tf.Graph().as_default() as graph:
        with tf.Session() as embed_sess:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))
            facenet.load_model(args.model_dir, meta_file, ckpt_file)
            images_placeholder = graph.get_tensor_by_name("input:0")
            embeddings = graph.get_tensor_by_name("embeddings:0")
            phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")

            with tf.Graph().as_default():
                detect_sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
                with detect_sess.as_default():
                    pnet, rnet, onet = detect_face.create_mtcnn(detect_sess, None)
                    print("Starting the API")
                    app.run(host='0.0.0.0')

