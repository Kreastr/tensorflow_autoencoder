from setuptools import setup

install_requires = [ 'tensorflow']

setup(name='tfautoencoder',
    version='0.1.2',
    author='Aleksei Romanenko',
    author_email='aleksei.romanenko@pm.me',
    platform='noarch',
    license='GPLv3',
    description='Tensorflow Model builder for general purpose Autoencoder. ',
    long_description="""Tensorflow Model builder for general purpose Autoencoder.""",
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
    url='https://github.com/Kreastr/tensorflow_autoencoder',
    packages=['tfautoencoder'],
    install_requires=install_requires
    )
