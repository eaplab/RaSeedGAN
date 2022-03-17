python -u run_generate_tfrecords.py -c pinball -u 4 -n 010
python -u run_training.py -c pinball -u 4 -m architecture-01-noise-010 -n 10
python run_compute_predictions.py -c pinball -u 4 -m architecture-01-noise-010 -n 10
python -u run_evaluate_predictions.py -c pinball -u 4 -m architecture-01-noise-010 -n 010