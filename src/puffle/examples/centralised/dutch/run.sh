# Experiment with centralised learning with the Dutch dataset
# Puffle is used to mitigate the unfairness using target 0.05
uv run python main.py --batch_size=805 --lr=0.04119584533428087 --optimizer=adam --regularization_lambda=0.8231482269709784 --project_name PuffleTest --target 0.05 --epochs 10 --regularization_mode fixed --csv_path ../../data/dutch/


# Experiment with centralised learning with the Dutch dataset
# Puffle is used to mitigate the unfairness using target 0.05
# In this case Tunable Lambda is used
uv run python main.py --batch_size=732 --lr=0.05840229525088357 --optimizer=adam --project_name PuffleTest --target 0.05 --epochs 10 --regularization_mode tunable --csv_path ../../data/dutch/ --alpha 1.9086195374853872 --momentum 0.8453489657333905 --weight_decay_alpha 0.7818180302292208
