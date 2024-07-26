PROJECT_NAME="PUFFLE_Celeba" # swap out globally

run_sweep_and_agent () {
  # Set the SWEEP_NAME variable
  SWEEP_NAME="$1"

  
  # Run the wandb sweep command and store the output in a temporary file
  poetry run wandb sweep --project "$PROJECT_NAME" --name "$SWEEP_NAME" "$SWEEP_NAME.yaml" >temp_output.txt 2>&1
  
  # Extract the sweep ID using awk
  SWEEP_ID=$(awk '/wandb agent/{ match($0, /wandb agent (.+)/, arr); print arr[1]; }' temp_output.txt)

  # Remove the temporary output file
#   rm temp_output.txts
  
  # Run the wandb agent command
  poetry run wandb agent $SWEEP_ID --project "$PROJECT_NAME" --count 20
}

run_sweep_and_agent "baseline_dp_5"
run_sweep_and_agent "tunable_dp_5_t_0.06"



# run_sweep_and_agent "baseline"

# run_sweep_and_agent "baseline_dp_8"
# run_sweep_and_agent "baseline_dp_5"

# run_sweep_and_agent "fixed_t_0.04"
# run_sweep_and_agent "fixed_t_0.06"
# run_sweep_and_agent "fixed_t_0.09"
# run_sweep_and_agent "fixed_t_0.12"
# run_sweep_and_agent "fixed_t_0.15"

# run_sweep_and_agent "fixed_dp_5_t_0.04"
# run_sweep_and_agent "fixed_dp_5_t_0.06"
# run_sweep_and_agent "fixed_dp_5_t_0.09"
# run_sweep_and_agent "fixed_dp_5_t_0.12"
# run_sweep_and_agent "fixed_dp_5_t_0.15"

# run_sweep_and_agent "fixed_dp_8_t_0.04"
# run_sweep_and_agent "fixed_dp_8_t_0.06"
# run_sweep_and_agent "fixed_dp_8_t_0.09"
# run_sweep_and_agent "fixed_dp_8_t_0.12"
# run_sweep_and_agent "fixed_dp_8_t_0.15"

# run_sweep_and_agent "tunable_t_0.04"
# run_sweep_and_agent "tunable_t_0.06"
# run_sweep_and_agent "tunable_t_0.09"
# run_sweep_and_agent "tunable_t_0.12"
# run_sweep_and_agent "tunable_t_0.15"

# run_sweep_and_agent "tunable_dp_5_t_0.04"
# run_sweep_and_agent "tunable_dp_5_t_0.06"
# run_sweep_and_agent "tunable_dp_5_t_0.09"
# run_sweep_and_agent "tunable_dp_5_t_0.12"
# run_sweep_and_agent "tunable_dp_5_t_0.15"

# run_sweep_and_agent "tunable_dp_8_t_0.04"
# run_sweep_and_agent "tunable_dp_8_t_0.06"
# run_sweep_and_agent "tunable_dp_8_t_0.09"
# run_sweep_and_agent "tunable_dp_8_t_0.12"
# run_sweep_and_agent "tunable_dp_8_t_0.15"