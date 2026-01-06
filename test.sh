export PYTHONPATH=.:$PYTHONPATH
python -m debugpy --connect 127.0.0.1:5678 --wait-for-client infer.py 