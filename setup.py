import setuptools

with open("README.md", "rb") as fh:
    long_description = fh.read().decode("utf-8")

setuptools.setup(
    name="vosk-tts",
    version="0.3.60",
    author="Alpha Cephei Inc",
    author_email="contact@alphacephei.com",
    description="Offline text to speech synthesis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alphacep/vosk-tts",
    packages=setuptools.find_packages(),
    entry_points = {
        'console_scripts': ['vosk-tts=vosk_tts.cli:main'],
    },
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    python_requires='>=3.7',
    install_requires=['onnxruntime>=1.14', 'tqdm', 'requests', 'tokenizers'],
)
