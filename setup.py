#!/usr/bin/env python

from distutils.core import setup

setup(name='facenet-service',
      version='0.1',
      description='Face Recognition as a service using Tensorflow',
      author='Youfie Technologies Inc',
      author_email='trevor@youfie.io',
      url='https://github.com/trevorprater/facial/',
      packages=['align', 'models', 'test'], )
