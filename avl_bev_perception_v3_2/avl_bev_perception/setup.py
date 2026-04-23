from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'avl_bev_perception'

setup(
    name=package_name,
    version='3.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'rviz'),   glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='AVL Team',
    maintainer_email='avl@example.com',
    description='IGVC AutoNav 3-camera ZED X BEV perception with hybrid '
                'HSV + ONNX semantic segmentation',
    license='MIT',
    entry_points={
        'console_scripts': [
            'bev_perception_node = avl_bev_perception.bev_perception_node:main',
        ],
    },
)
