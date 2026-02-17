PROJECT_NAME="TestPuffleFL" # swap out globally

run_sweep_and_agent () {
  # Set the SWEEP_NAME variable
  SWEEP_NAME="$1"

  
  # Run the wandb sweep command and store the output in a temporary file
  uv run wandb sweep --project "$PROJECT_NAME" --name "$SWEEP_NAME" "$SWEEP_NAME.yaml" >temp_output.txt 2>&1
  
  # Extract the sweep ID using awk
  SWEEP_ID=$(awk '/wandb agent/{ match($0, /wandb agent (.+)/, arr); print arr[1]; }' temp_output.txt)

  # Remove the temporary output file
#   rm temp_output.txt
  
  # Run the wandb agent command
  uv run wandb agent $SWEEP_ID --project "$PROJECT_NAME" --count 30
}

# run_sweep_and_agent "baseline_dutch_cross_device"
# run_sweep_and_agent "private_baseline_dutch_cross_device_epsilon_0.5"
# run_sweep_and_agent "private_baseline_dutch_cross_device_epsilon_1.0"

# run_sweep_and_agent "private_tunable_dutch_cross_device"
# run_sweep_and_agent "tunable_dutch_cross_device"

# run_sweep_and_agent "private_fixed_dutch_cross_device"
# run_sweep_and_agent "fixed_dutch_cross_device"

# run_sweep_and_agent "private_fixed_dutch_cross_device_epsilon_0.5_target_10"
# run_sweep_and_agent "private_fixed_dutch_cross_device_epsilon_0.5_target_25"
# run_sweep_and_agent "private_fixed_dutch_cross_device_epsilon_0.5_target_50"
# run_sweep_and_agent "private_fixed_dutch_cross_device_epsilon_0.5_target_65"
# run_sweep_and_agent "private_fixed_dutch_cross_device_epsilon_0.5_target_75"

# run_sweep_and_agent "private_fixed_dutch_cross_device_epsilon_1.0_target_10"
# run_sweep_and_agent "private_fixed_dutch_cross_device_epsilon_1.0_target_25"
# run_sweep_and_agent "private_fixed_dutch_cross_device_epsilon_1.0_target_50"
run_sweep_and_agent "private_fixed_dutch_cross_device_epsilon_1.0_target_65"
run_sweep_and_agent "private_fixed_dutch_cross_device_epsilon_1.0_target_75"

# run_sweep_and_agent "private_tunable_dutch_cross_device_epsilon_0.5_target_10"
# run_sweep_and_agent "private_tunable_dutch_cross_device_epsilon_0.5_target_25"
# run_sweep_and_agent "private_tunable_dutch_cross_device_epsilon_0.5_target_50"
# run_sweep_and_agent "private_tunable_dutch_cross_device_epsilon_0.5_target_65"
# run_sweep_and_agent "private_tunable_dutch_cross_device_epsilon_0.5_target_75"

# run_sweep_and_agent "private_tunable_dutch_cross_device_epsilon_1.0_target_10"
# run_sweep_and_agent "private_tunable_dutch_cross_device_epsilon_1.0_target_25"
# run_sweep_and_agent "private_tunable_dutch_cross_device_epsilon_1.0_target_50"
# run_sweep_and_agent "private_tunable_dutch_cross_device_epsilon_1.0_target_65"
# run_sweep_and_agent "private_tunable_dutch_cross_device_epsilon_1.0_target_75"

