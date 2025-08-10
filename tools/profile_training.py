#!/usr/bin/env python3
"""
Simple PyTorch Profiler for training analysis
"""

import torch
import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity
import os
import sys

def profile_training():
    """Profile the training process using PyTorch built-in profiler"""
    
    print("Starting PyTorch Profiler...")
    
    # Start profiling
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,        # Wait for 1 step
            warmup=1,      # Warmup for 1 step  
            active=10,     # Profile for 10 steps
            repeat=1       # Repeat once
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_flops=True,
        with_modules=True
    ) as prof:
        
        print("Profiler started. Now run your training command:")
        print("cd /workspace/SparseDrive && conda run -n sparsedrive bash -lc 'PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_train.sh projects/configs/sparsedrive_small_stage1.py 1 --deterministic'")
        
        # Wait for user to start training
        input("Press Enter after starting training to continue...")
        
        # Keep profiler running
        try:
            while True:
                prof.step()
                print(f"Profiler step {prof.step_num}")
        except KeyboardInterrupt:
            print("\nProfiling stopped by user")
        
        # Print summary
        print("\n=== Profiler Summary ===")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        
        # Save detailed results
        prof.export_stacks("./profiler_stacks.txt")
        print(f"\nDetailed stacks saved to: ./profiler_stacks.txt")
        print(f"TensorBoard logs saved to: ./profiler_logs/")
        print(f"Use 'tensorboard --logdir=./profiler_logs' to view results")

if __name__ == "__main__":
    profile_training()
