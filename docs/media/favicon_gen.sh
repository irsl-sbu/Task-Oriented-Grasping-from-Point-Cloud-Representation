#!/bin/bash

echo "Deleting existing images..."
rm 16.png 32.png 48.png favicon.ico

echo "Generating new images..."
inkscape -w 16 -h 16 -o 16.png favicon.svg
inkscape -w 32 -h 32 -o 32.png favicon.svg
inkscape -w 48 -h 48 -o 48.png favicon.svg

echo "Generating favicon.ico..."
convert 16.png 32.png 48.png favicon.ico

echo "Finished generating favicon.ico!"
