from setuptools import find_packages, setup
from glob import glob

package_name = "so101_perception"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/rviz",   glob("rviz/*.rviz")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Hector Tovar",
    maintainer_email="h.tovarm07@gmail.com",
    description="Colour-based 3-D object detection with RealSense D435",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "blue_object_detector = so101_perception.blue_object_detector:main",
            "hsv_calibrator       = so101_perception.hsv_calibrator:main",
            "object_classifier    = so101_perception.object_classifier:main",
        ],
    },
)
