#!/bin/bash
set -e

date
python train.py --stratified --fold 0
python train.py --stratified --fold 1
python train.py --stratified --fold 2
python train.py --stratified --fold 3
python train.py --stratified --fold 4
date
