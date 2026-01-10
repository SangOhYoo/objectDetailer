import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from core.detail_daemon import DetailDaemonContext

# Mock Scheduler
class MockScheduler:
    def __init__(self):
        self.timesteps = torch.tensor([900, 800, 700, 600], dtype=torch.long)
        self.sigmas = torch.tensor([10.0, 5.0, 2.0, 0.5, 0.0], dtype=torch.float32)
        self.step_called = 0
        
    def scale_model_input(self, sample, timestep):
        return sample 
        
    def step(self, model_output, timestep, sample, **kwargs):
        self.step_called += 1
        return MagicMock(prev_sample=sample)

class MockPipeline:
    def __init__(self):
        self.scheduler = MockScheduler()
        self.guidance_scale = 7.5

def test_dd_mechanism():
    pipe = MockPipeline()
    config = {
        'mode': 'both',
        'amount': 0.1, 
        'start': 0.0,
        'end': 1.0,
        'bias': 0.5,
        'exponent': 1.0,
        'start_offset': 0.5,
        'end_offset': 0,
        'fade': 0,
        'smooth': False
    }
    
    # Enable
    enabled = True
    
    with DetailDaemonContext(pipe, enabled, config) as dd:
        # Simulate Loop
        # Step 0
        t = pipe.scheduler.timesteps[0] # 900
        
        # Check Initial Sigma
        print(f"Initial Sigma[0]: {pipe.scheduler.sigmas[0]}")
        
        # 1. Scale Input
        # This should trigger patch
        sample = torch.randn(1, 4, 64, 64)
        _ = pipe.scheduler.scale_model_input(sample, t)
        
        # Check Sigma Modified
        modified_sigma = pipe.scheduler.sigmas[0].item()
        print(f"Modified Sigma[0]: {modified_sigma}")
        
        # Expected:
        # With sensitivity 0.25 and start_offset 0.5:
        # m = 0.5 * 0.25 = 0.125
        # factor = 1.0 - (0.125 * 7.5) = 0.0625
        # New Sigma = 10.0 * 0.0625 = 0.625
        
        print(f"Modified Sigma[0]: {modified_sigma}")
        assert modified_sigma < 1.0
        
        # 2. Step
        # This should restore Sigma
        pipe.scheduler.step(sample, t, sample)
        
        restored_sigma = pipe.scheduler.sigmas[0].item()
        print(f"Restored Sigma[0]: {restored_sigma}")
        
        assert restored_sigma == 10.0

if __name__ == "__main__":
    test_dd_mechanism()
