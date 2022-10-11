python -u run_training.py -c pinball -u 2 -m architecture-01-noise-010 -n 10
python -u run_training.py -c pinball -u 4 -m architecture-01-noise-010 -n 10
python -u run_training.py -c pinball -u 8 -m architecture-01-noise-010 -n 10

python -u run_training.py -c pinball -u 2 -m architecture-01-noise-050 -n 50
python -u run_training.py -c pinball -u 4 -m architecture-01-noise-050 -n 50
python -u run_training.py -c pinball -u 8 -m architecture-01-noise-050 -n 50

python -u run_training.py -c channel -u 2 -m architecture-01-noise-010 -n 10
python -u run_training.py -c channel -u 4 -m architecture-01-noise-010 -n 10
python -u run_training.py -c channel -u 8 -m architecture-01-noise-010 -n 10

python -u run_training.py -c sst -u 2 -m architecture-02-noise-000 -n 0
python -u run_training.py -c sst -u 4 -m architecture-02-noise-000 -n 0
python -u run_training.py -c sst -u 8 -m architecture-02-noise-000 -n 0

python -u run_training.py -c exptbl -u 2 -m architecture-01-noise-000 -n 0
python -u run_training.py -c exptbl -u 4 -m architecture-01-noise-000 -n 0
