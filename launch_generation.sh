python -u run_generate_tfrecords.py -c pinball -u 2 -n 010
python -u run_generate_tfrecords.py -c pinball -u 4 -n 010
python -u run_generate_tfrecords.py -c pinball -u 8 -n 010

python -u run_generate_tfrecords.py -c pinball -u 2 -n 050
python -u run_generate_tfrecords.py -c pinball -u 4 -n 050
python -u run_generate_tfrecords.py -c pinball -u 8 -n 050

python -u run_generate_tfrecords.py -c channel -u 2 -n 010
python -u run_generate_tfrecords.py -c channel -u 4 -n 010
python -u run_generate_tfrecords.py -c channel -u 8 -n 010

python -u run_generate_tfrecords.py -c sst -u 2 -n 0
python -u run_generate_tfrecords.py -c sst -u 4 -n 0
python -u run_generate_tfrecords.py -c sst -u 8 -n 0

python -u run_generate_tfrecords.py -c exptbl -u 2 -n 0
python -u run_generate_tfrecords.py -c exptbl -u 4 -n 0
