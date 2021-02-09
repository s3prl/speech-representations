import sys
import argparse

from speech_reps.featurize import DeCoARFeaturizer

parser = argparse.ArgumentParser()
parser.add_argument('--input_wav', required=True)
args = parser.parse_args()

# Load the model on GPU 0
featurizer = DeCoARFeaturizer('artifacts/decoar-encoder-29b8e2ac.params', gpu=0)
# Returns a (time, feature) NumPy array
data = featurizer.file_to_feats(args.input_wav)
