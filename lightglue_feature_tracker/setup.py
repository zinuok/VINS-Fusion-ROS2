from setuptools import setup, find_packages
from glob import glob

package_name = 'lightglue_feature_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[
        'lightglue_feature_tracker',
        'lightglue_feature_tracker.*'
    ]),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),  # Utilisation de glob pour récupérer tous les fichiers .launch.py
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/lightglue', glob('lightglue_feature_tracker/lightglue/*.py')), 
    ],
    install_requires=[
        'setuptools',
        'torch',
        'scipy',
        'matplotlib',
        'numpy',
        'opencv-python'
    ],
    zip_safe=True,
    maintainer='root',
    maintainer_email='jomimassi@gmail.com',
    description='Feature tracker using LightGlue in ROS2.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'feature_tracker_node = lightglue_feature_tracker.feature_tracker_node:main',
        ],
    },
)
