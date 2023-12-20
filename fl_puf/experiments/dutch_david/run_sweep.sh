PROJECT_NAME="FL_PUF_Sweep" # swap out globally

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
  poetry run wandb agent $SWEEP_ID --project "$PROJECT_NAME" --count 25
}

# run_sweep_and_agent "baseline_private"
# run_sweep_and_agent "tunable_private"
# run_sweep_and_agent "fixed_private"

run_sweep_and_agent "tunable_private_weighted" 
run_sweep_and_agent "fixed_private_weighted" 

run_sweep_and_agent "tunable_private_1_weighted" 
run_sweep_and_agent "fixed_private_1_weighted" 

# run_sweep_and_agent "tunable_private_target_005"
# run_sweep_and_agent "fixed_private_target_005"

run_sweep_and_agent "tunable_private_weighted_target_005" 
run_sweep_and_agent "fixed_private_weighted_target_005" 
 
run_sweep_and_agent "tunable_private_1_weighted_target_005" 
run_sweep_and_agent "fixed_private_1_weighted_target_005" 

# run_sweep_and_agent "tunable_private_target_008"
# run_sweep_and_agent "fixed_private_target_008"

run_sweep_and_agent "tunable_private_weighted_target_008" 
run_sweep_and_agent "fixed_private_weighted_target_008" 

run_sweep_and_agent "tunable_private_1_weighted_target_008" 
run_sweep_and_agent "fixed_private_1_weighted_target_008" 

# run_sweep_and_agent "fixed_private"
# run_sweep_and_agent "baseline_private"
# run_sweep_and_agent "fixed_private"
# run_sweep_and_agent "tunable_private"
# run_sweep_and_agent "fixed_private"
# run_sweep_and_agent "tunable_private"
# run_sweep_and_agent "fixed_private"
# run_sweep_and_agent "tunable_private"
# run_sweep_and_agent "fixed_private"
# run_sweep_and_agent "tunable_private"
# run_sweep_and_agent "fixed_private"
# run_sweep_and_agent "tunable_private"
# run_sweep_and_agent "fixed_private"
# run_sweep_and_agent "tunable_private"
# run_sweep_and_agent "fixed_private"
# run_sweep_and_agent "tunable_private"
# run_sweep_and_agent "fixed_private"
# run_sweep_and_agent "tunable_private"
# run_sweep_and_agent "fixed_private"
# run_sweep_and_agent "tunable_private"

