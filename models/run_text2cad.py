"""
Text2CAD (SadilKhan/Text2CAD) inference for TextCAD-Bench.
Uses the official Text2CAD checkpoint and generates DeepCAD token sequences,
converted to mesh via the Text2CAD pipeline.

Usage:
    python models/run_text2cad.py --test_set T1 \
        --data data/t1_samples.json \
        --gt_dir data/gt_meshes \
        --output results/text2cad_t1_results.json

NOTE: Text2CAD requires its own environment (see Text2CAD repo for setup).
Clone https://github.com/SadilKhan/Text2CAD and follow their installation
instructions, then run this script from within that environment.
"""
import argparse, json, os, sys
print("Text2CAD requires the official Text2CAD repo environment.")
print("See: https://github.com/SadilKhan/Text2CAD")
print("Run their inference script, then use eval/evaluate.py on the outputs.")
