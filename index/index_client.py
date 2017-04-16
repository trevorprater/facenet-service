import sys
import uuid
from pprint import pprint

import fire
import psycopg2
import psycopg2.extras
from annoy import AnnoyIndex


class IndexClient(object):
    def __init__(self):
        self.db_uri = "postgres://postgres@localhost:5432/facenet"
        self.conn = psycopg2.connect(
            self.db_uri, cursor_factory=psycopg2.extras.RealDictCursor)

    def index_faces(self, num_faces, num_trees=500, dimensionality=128):
        sql = "select faces.id, photos.url, faces.feature_vector, faces.photo_id FROM faces LEFT JOIN photos ON \
                photos.id=faces.photo_id LIMIT {}".format(
            num_faces)

        cursor = self.conn.cursor(
            'crsr', cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(sql)
        aindex = AnnoyIndex(dimensionality)
        mappings = []
        for ctr, face in enumerate(cursor):
            if ctr % 10000 == 0:
                print 'read 10,000 rows'
            aindex.add_item(ctr, face['feature_vector'])
            mappings.append(face['id'])
        cursor.close()
        cursor = self.conn.cursor()

        print 'rows added to index. building index...'
        aindex.build(num_trees)
        print 'index built. saving index...'
        ndx_uuid = str(uuid.uuid4())
        aindex.save('{}.ann'.format(ndx_uuid))
        print 'index saved to {}.ann'.format(ndx_uuid)

        sql = "INSERT INTO ndx_mappings(id, face_id, index_id) VALUES ('{}', '{}', '{}')"
        for ctr, face_id in enumerate(mappings):
            cursor.execute(sql.format(ndx_uuid, face_id, ctr))
        self.conn.commit()
        cursor.close()
        self.conn.close()

    def search_index(self,
                     filename,
                     face_id,
                     num_neighbors=100,
                     search_k=-1,
                     dimensionality=128):
        ndx_uuid = filename.replace('.ann', '')
        u = AnnoyIndex(dimensionality)
        print 'loading index'
        u.load(filename)
        print 'done loading index'

        query_ctr = 0
        while True:
            # to allow for successive queries w/o reloading the index from disk
            if query_ctr > 0:
                print '\n please enter a face_id to query'
                face_id = sys.stdin.readline().strip()

            sql = "select index_id from ndx_mappings where id = '{}' and face_id = '{}';".format(
                ndx_uuid, face_id)
            cursor = self.conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(sql)
            index_id = int(cursor.fetchone()['index_id'])
            cursor.close()
            results, distances = u.get_nns_by_item(
                index_id,
                num_neighbors,
                search_k=search_k,
                include_distances=True)

            cursor = self.conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor)
            output = []
            sql = "SELECT face_id FROM ndx_mappings WHERE id = '{}' AND index_id = '{}'"
            photosql = "SELECT photo_id FROM faces where id = '{}'"
            urlsql = "SELECT url FROM photos WHERE id = '{}'"
            for ctr, res in enumerate(results):
                cursor.execute(sql.format(ndx_uuid, res))
                face_id = cursor.fetchone()['face_id']
                cursor.execute(photosql.format(face_id))
                photo_id = cursor.fetchone()['photo_id']
                cursor.execute(urlsql.format(photo_id))
                url = cursor.fetchone()['url']
                print '{}, dist = {}'.format(url, distances[ctr])
            cursor.close()
            query_ctr += 1


if __name__ == '__main__':
    fire.Fire(IndexClient)
