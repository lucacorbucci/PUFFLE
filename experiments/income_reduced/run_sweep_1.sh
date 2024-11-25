PROJECT_NAME="Multi_fairness" # swap out globally

run_sweep_and_agent () {
  # Set the SWEEP_NAME variable
  SWEEP_NAME="$1"

  
  # Run the wandb sweep command and store the output in a temporary file
  poetry run wandb sweep --project "$PROJECT_NAME" --name "$SWEEP_NAME" "$SWEEP_NAME.yaml" >temp_output.txt 2>&1
  
  # Extract the sweep ID using awk
  SWEEP_ID=$(awk '/wandb agent/{ match($0, /wandb agent (.+)/, arr); print arr[1]; }' temp_output.txt)

  # Remove the temporary output file
  rm temp_output.txt
  
  # Run the wandb agent command
  poetry run wandb agent $SWEEP_ID --project "$PROJECT_NAME" --count 20
}

run_sweep_and_agent "baseline"
run_sweep_and_agent "005"
run_sweep_and_agent "010"


# /usr/bin/env poetry run python ../../puffle/main.py --batch_size=1946 --epochs=1 --lr=0.06797031471808428 --optimizer=adam --dataset income --num_rounds 10 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 10 --sampled_clients 1.0 --sampled_clients_test 0 --sampled_clients_validation 1.0 --debug False --base_path ../../../reduced_income_data/ --dataset_path ../../../reduced_income_data/ --seed 41 --wandb True --sweep True --training_nodes 0.61 --validation_nodes 0.2 --test_nodes 0.2 --tabular_data True --one_group_nodes True --update_lambda False --metric disparity --splitted_data_dir federated