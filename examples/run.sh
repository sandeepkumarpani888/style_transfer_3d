#!/usr/bin/env bash

DIR=./examples/data

# To test out with your own examples here are the steps:
# Put your file in /examples/data under style_transfer_3d
# Lets say the obj file which you put is called architecture.obj
# The change: {ME} in the line below everywhere with architecture
# The generated gif will be avilable as /examples/data/architeture.gif
########################
python ./examples/run.py -im ${DIR}/house_10.obj -is ${DIR}/styles/gris1.jpg -o ${DIR}/house_10.gif -lc 2000000000 -ltv 10000
########################
# python ./examples/run.py -im ${DIR}/meshes/teapot.obj -is ${DIR}/styles/gris1.jpg -o ${DIR}/results/teapot_gris.gif -lc 2000000000 -ltv 10000
# python ./examples/run.py -im ${DIR}/meshes/bunny.obj -is ${DIR}/styles/munch1.jpg -o ${DIR}/results/bunny_munch.gif -lc 2000000000 -ltv 100000
