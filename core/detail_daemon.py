import numpy as np
from contextlib import contextmanager

def make_schedule(steps, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth):
    """
    Calculates the Detail Daemon noise schedule multipliers.
    """
    if steps <= 0:
        return np.zeros(0)
        
    start = min(start, end)
    mid = start + bias * (end - start)
    multipliers = np.zeros(steps)

    start_idx, mid_idx, end_idx = [int(round(x * (steps - 1))) for x in [start, mid, end]]            

    # Start to Mid
    if mid_idx >= start_idx:
        start_values = np.linspace(0, 1, mid_idx - start_idx + 1)
        if smooth:  
            start_values = 0.5 * (1 - np.cos(start_values * np.pi))
        start_values = start_values ** exponent
        if start_values.any():
            start_values *= (amount - start_offset)  
            start_values += start_offset
        multipliers[start_idx:mid_idx+1] = start_values

    # Mid to End
    if end_idx >= mid_idx:
        end_values = np.linspace(1, 0, end_idx - mid_idx + 1)
        if smooth:
            end_values = 0.5 * (1 - np.cos(end_values * np.pi))
        end_values = end_values ** exponent
        if end_values.any():
            end_values *= (amount - end_offset)  
            end_values += end_offset  
        multipliers[mid_idx:end_idx+1] = end_values        

    # Offsets and Fade
    multipliers[:start_idx] = start_offset
    multipliers[end_idx+1:] = end_offset    
    multipliers *= 1 - fade

    return multipliers

class DetailDaemonContext:
    """
    Context manager to patch the pipe's scheduler step function to inject noise.
    """
    def __init__(self, pipe, enabled, config):
        import torch
        self.pipe = pipe
        self.enabled = enabled
        self.config = config
        self.original_step = None
        self.multipliers = None
        self.current_step_idx = 0
        self.total_steps = 0
        self.batch_size = 1
        self.last_modified_idx = -1

    def __enter__(self):
        # [Debug]
        print(f"    [DetailDaemon] Entering Context. Enabled: {self.enabled}")
        
        # Capture scheduler
        self.scheduler = self.pipe.scheduler
        self.original_step = self.scheduler.step
        
        # Monkey patch step
        if self.enabled:
            self.original_scale_model_input = self.scheduler.scale_model_input
            self.scheduler.scale_model_input = self.patched_scale_model_input
            self.current_step_idx = 0
            
            if hasattr(self.scheduler, "sigmas"):
                print(f"    [DetailDaemon] Scheduler Sigmas shape: {self.scheduler.sigmas.shape}")
        
        return self

    def patched_scale_model_input(self, sample, timestep):
        if self.multipliers is None:
            self._init_multipliers()

        idx = self.current_step_idx
        
        if self.multipliers is not None and idx < len(self.multipliers):
            if self.last_modified_idx == idx:
                return self.original_scale_model_input(sample, timestep)
            
            m = self.multipliers[idx] * 0.25
            
            cfg_scale = getattr(self.pipe, 'guidance_scale', 7.5)
            mode = self.config.get('mode', 'both')
            factor = 1.0
            if mode == 'cond': factor = 1.0 - m
            elif mode == 'uncond': factor = 1.0 + m
            else: factor = 1.0 - (m * cfg_scale)
            
            factor = max(0.01, factor)

            if idx < len(self.scheduler.sigmas):
                original_sigma = self.scheduler.sigmas[idx].item()
                new_sigma = original_sigma * factor
                self.scheduler.sigmas[idx] = new_sigma
                
                self.saved_sigma_idx = idx
                self.saved_sigma_val = original_sigma
                self.last_modified_idx = idx
                
                return self.original_scale_model_input(sample, timestep)
                
        return self.original_scale_model_input(sample, timestep)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled:
            # Restore
            if self.original_step:
                self.pipe.scheduler.step = self.original_step
            if hasattr(self, 'original_scale_model_input'):
                self.pipe.scheduler.scale_model_input = self.original_scale_model_input
            
            # Reset
            self.multipliers = None

    def _init_multipliers(self):
        # We need total steps.
        # pipe.scheduler.timesteps should be populated.
        timesteps = self.pipe.scheduler.timesteps
        if timesteps is None or len(timesteps) == 0:
             return
        
        total_steps = len(timesteps)
        self.multipliers = make_schedule(
            total_steps,
            self.config.get('start', 0.2),
            self.config.get('end', 0.8),
            self.config.get('bias', 0.5),
            self.config.get('amount', 0.1),
            self.config.get('exponent', 1.0),
            self.config.get('start_offset', 0.0),
            self.config.get('end_offset', 0.0),
            self.config.get('fade', 0.0),
            self.config.get('smooth', False)
        )
        self.total_steps = total_steps
        
        # Monkey patch 'step' as well to increment index and restore sigma
        if self.original_step is None: # Should be captured in __enter__
             pass
        
        # We replace step here if not done
        if self.pipe.scheduler.step == self.original_step:
             self.pipe.scheduler.step = self.patched_step

    def patched_step(self, *args, **kwargs):
        # Call original
        res = self.original_step(*args, **kwargs)
        
        # Restore sigma if we modified it
        if hasattr(self, 'saved_sigma_idx'):
            self.scheduler.sigmas[self.saved_sigma_idx] = self.saved_sigma_val
            del self.saved_sigma_idx
            del self.saved_sigma_val
            
        self.current_step_idx += 1
        return res
