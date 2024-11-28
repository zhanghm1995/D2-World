#!/bin/bash

echo "================Extract all LiDAR files================"
for i in {0..24}
do
    filename="openscene_sensor_trainval_lidar_${i}.tgz"
    echo "Extracting ${filename}..."
    tar -xzf openscene-v1.1/${filename}
done
