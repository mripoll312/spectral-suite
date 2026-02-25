import setuptools

setuptools.setup(
    name='gen5_reader',
    version='0.1.3',
    url='',

    author='Robert Giessmann',
    author_email='r.giessmann@tu-berlin.de',

    description='',
    long_description=open('README.md').read(),

    packages=setuptools.find_packages(exclude=['test', 'test.*']),

    platforms='any',

    include_package_data=False,
    package_data = {
    },

    install_requires=[
        'pandas',
    ],

    entry_points={
        'console_scripts' : [
        'gen5split=gen5_reader.split:main',
        'gen5merge=gen5_reader.merge:main'
        ],
    },

    classifiers = [],
)
