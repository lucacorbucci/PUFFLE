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
  poetry run wandb agent $SWEEP_ID --project "$PROJECT_NAME" --count 15
}

# run_sweep_and_agent "005_fixed"
# run_sweep_and_agent "005_tunable"
# run_sweep_and_agent "01_fixed"
run_sweep_and_agent "01_tunable"
run_sweep_and_agent "015_fixed"
run_sweep_and_agent "015_tunable"
run_sweep_and_agent "02_fixed"
run_sweep_and_agent "02_tunable"
run_sweep_and_agent "025_fixed"
run_sweep_and_agent "025_tunable"
run_sweep_and_agent "03_fixed"
run_sweep_and_agent "03_tunable"
run_sweep_and_agent "04_fixed"
run_sweep_and_agent "04_tunable"