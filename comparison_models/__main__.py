"""Display model architecture or counts; never train or access a dataset."""

import argparse
import json

import tensorflow as tf

from . import MODEL_BUILDERS, build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all"] + list(MODEL_BUILDERS), default="all")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    names = list(MODEL_BUILDERS) if args.model == "all" else [args.model]
    rows = []
    for name in names:
        tf.keras.backend.clear_session()
        model = build_model(name)
        if args.summary:
            model.summary()
        rows.append({
            "model": name, "input_shape": model.input_shape,
            "output_shape": model.output_shape, "total_params": model.count_params(),
            "trainable_params": sum(int(tf.size(w)) for w in model.trainable_weights),
            "non_trainable_params": sum(int(tf.size(w)) for w in model.non_trainable_weights),
        })
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
