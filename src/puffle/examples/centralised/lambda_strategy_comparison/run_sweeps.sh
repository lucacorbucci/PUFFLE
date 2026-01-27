#!/bin/bash

# Run WandB sweeps for hyperparameter tuning of all three strategies

echo "Starting hyperparameter tuning sweeps..."
echo ""
echo "Run these commands in separate terminals:"
echo ""
echo "Terminal 1 (Momentum):"
echo "  cd src/puffle/examples/centralised/lambda_strategy_comparison"
echo "  wandb sweep momentum_strategy.yaml"
echo "  wandb agent <sweep-id-from-above>"
echo ""
echo "Terminal 2 (Gradient):"
echo "  cd src/puffle/examples/centralised/lambda_strategy_comparison"
echo "  wandb sweep gradient_strategy.yaml"
echo "  wandb agent <sweep-id-from-above>"
echo ""
echo "Terminal 3 (PID):"
echo "  cd src/puffle/examples/centralised/lambda_strategy_comparison"
echo "  wandb sweep pid_strategy.yaml"
echo "  wandb agent <sweep-id-from-above>"
