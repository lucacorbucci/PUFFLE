# Experiment with centralised learning with the Dutch dataset
# Puffle is used to mitigate the unfairness using target 0.05
uv run python main.py --batch_size=675 --lr=0.08747688288865933 --optimizer=sgd --regularization_lambda=0.8640987672519225 --project_name PuffleTest --target 0.05 --epochs 10 --regularization_mode fixed --csv_path ../../data/dutch/