PROJECT_NAME="Celeba_error_rate_40" # swap out globally

run_sweep_and_agent () {
  # Set the SWEEP_NAME variable
  SWEEP_NAME="$1"

  
  # Run the wandb sweep command and store the output in a temporary file
  poetry run wandb sweep --project "$PROJECT_NAME" --name "$SWEEP_NAME" "$SWEEP_NAME.yaml" >temp_output.txt 2>&1
  
  # Extract the sweep ID using awk
  SWEEP_ID=$(awk '/wandb agent/{ match($0, /wandb agent (.+)/, arr); print arr[1]; }' temp_output.txt)

  # Remove the temporary output file
#   rm temp_output.txt
  
  # Run the wandb agent command
  poetry run wandb agent $SWEEP_ID --project "$PROJECT_NAME" --count 100
}

# run_sweep_and_agent "fixed_011_no_DP"



# run_sweep_and_agent "tunable_0.08_epsilon_8"
# run_sweep_and_agent "tunable_0.08_epsilon_10"

# run_sweep_and_agent "tunable_0.11_epsilon_8"
# run_sweep_and_agent "tunable_0.11_epsilon_10"

# run_sweep_and_agent "tunable_0.05_epsilon_8"

# run_sweep_and_agent "tunable_0.08_epsilon_8"
# run_sweep_and_agent "tunable_0.11_epsilon_8"

# run_sweep_and_agent "tunable_0.05_epsilon_10"
# run_sweep_and_agent "tunable_0.08_epsilon_10"
# run_sweep_and_agent "tunable_0.11_epsilon_10"

# run_sweep_and_agent "tunable_0.08_epsilon_10"
# 

# run_sweep_and_agent "tunable_0.17"
run_sweep_and_agent "tunable_0.20"
