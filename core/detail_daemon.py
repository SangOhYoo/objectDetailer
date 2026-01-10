import numpy as np
import torch
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
        self.pipe = pipe
        self.enabled = enabled
        self.config = config
        self.original_step = None
        self.multipliers = None
        self.current_step_idx = 0
        self.total_steps = 0
        self.batch_size = 1

    def __enter__(self):
        if not self.enabled:
            return self

        # Capture scheduler
        self.scheduler = self.pipe.scheduler
        self.original_step = self.scheduler.step
        
        # Monkey patch step
        # Note: We need to bind self to the patched method
        def patched_step(model_output, timestep, sample, **kwargs):
            # 1. Determine step index
            # This is tricky because schedulers diff in how they track steps.
            # Usually they have `self.step_index` or we can track it manually if we reset it.
            # Or we can infer from timestep if we know the schedule.
            
            # For simplicity, we track locally assuming sequential calls.
            # But we need to know Total Steps beforehand.
            
            # Calculate multipliers if not ready (Needs total steps)
            # Usually total steps is set in pipe before inference or passed to loop.
            # We assume total_steps is available or we can guess.
            
            # Access sigma
            # Most Diffusers schedulers store sigmas in `self.sigmas`.
            # Timestep `t` corresponds to `sigmas[t]`.
            
            # EXECUTE ORIGINAL STEP FIRST to get next sample (or we modify sigma before?)
            # Detail Daemon modifies 'params.sigma' BEFORE the step in A1111.
            # In Diffusers, 'step' takes 'model_output' (noise prediction) and 'sample' (current latent).
            # And it uses `self.sigmas` inside.
            
            # We need to modify `self.scheduler.sigmas` in place?
            # Or modify `sample`?
            # Detail Daemon logic:
            # params.sigma[i] *= 1 - multiplier (or + multiplier)
            
            # In Diffusers, `sigmas` is a tensor.
            # We can modify `self.scheduler.sigmas` corresponding to current timestep.
            
            # Let's find the current sigma index.
            # Timestep is a tensor or int.
            step_index = self.current_step_idx
            
            if self.multipliers is not None and step_index < len(self.multipliers):
                # [Fix] Increased sensitivity from 0.1 to 0.25 for better visibility in Inpainting
                m = self.multipliers[step_index] * 0.25 
                
                # Apply to Sigma
                # We need to find which sigma corresponds to this step.
                # In standard schedulers, `sigmas[step_index]` is used.
                
                # Mode check
                mode = self.config.get('mode', 'both')
                
                # In A1111:
                # cond -> sigma *= 1 - m
                # uncond -> sigma *= 1 + m
                # both -> sigma *= 1 - m * cfg_scale
                
                cfg_scale = getattr(self.pipe, 'guidance_scale', 7.5)
                factor = 1.0
                
                if mode == 'cond':
                    factor = 1.0 - m
                elif mode == 'uncond':
                    factor = 1.0 + m
                else: # both
                    factor = 1.0 - (m * cfg_scale)
                
                # Safety Clamp to prevent negative sigma
                factor = max(0.01, factor)

                if idx < len(self.scheduler.sigmas):
                    original_sigma = self.scheduler.sigmas[idx].item()
                    
                    # Modify
                    new_sigma = original_sigma * factor
                    self.scheduler.sigmas[idx] = new_sigma
        
        # We need a better hook. `step` is called at the END of the step.
        # But we need to modify sigma BEFORE `step` (or during noise prediction provided to UNet).
        # A1111 `on_cfg_denoiser` happens *inside* the sampler loop, before or during denoising.
        
        # In Diffusers pipeline:
        # for t in timesteps:
        #   latent_model_input = scheduler.scale_model_input(latents, t)  <-- Uses sigma
        #   noise_pred = unet(latent_model_input, t, ...)
        #   latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        # Only `scale_model_input` uses sigma to scale the input (for K-Diffusion/EDM).
        # And `step` uses sigma to calculate the next step.
        
        # If we change sigma in `scheduler`, both are affected. This seems correct.
        
        # We will wrap `scheduler.scale_model_input` and `scheduler.step`.
        
        self.original_scale_model_input = self.scheduler.scale_model_input
        self.scheduler.scale_model_input = self.patched_scale_model_input
        
        # We also need to reset step index on enter.
        # We also need to reset step index on enter.
        self.current_step_idx = 0
        
        return self

    def patched_scale_model_input(self, sample, timestep):
        # Calculate multipliers if needed
        if self.multipliers is None:
            self._init_multipliers()

        # Apply modification
        idx = self.current_step_idx
        
        if self.multipliers is not None and idx < len(self.multipliers):
             # [Fix] Sensitivity adjusted in patched_step logic above?
             # Wait, the logic was duplicated or moved?
             # I modified 'patched_step' above, but previously I inserted logic in 'patched_scale_model_input' during verification.
             # I need to clean up 'patched_scale_model_input' to NOT duplicate logic or conflicts.
             # DetailDaemon usually modifies sigma BEFORE step.
             # 'scale_model_input' happens before UNet.
             # So modification here is Correct.
             # The modification in 'patched_step' above (in previous replace content) was incorrect because 'patched_step' logic is after 'scale_model_input'.
             # Actually, I replaced lines 105-147 which is 'patched_step' logic inside __enter__.
             # BUT 'patched_step' wraps `step`.
             # `scale_model_input` wraps `scale_model_input`.
             # WE NEED TO MODIFY SIGMA IN ONE PLACE.
             # My verification test confirmed `patched_scale_model_input` is the place.
             # The previous 'replace' targeted lines 105-147 which is `patched_step`.
             # I accidentally put the SIGMA modification logic into `patched_step`?
             # No, lines 105-147 in original file were comments inside `patched_step`.
             # I put code there in previous turn? No, I edited `patched_step` to include logic?
             # In verify step, I edited `patched_scale_model_input` (lines 189...).
             
             # CRITICAL: I must put the logic in `patched_scale_model_input` (Line 189+), NOT `patched_step`.
             # I made a mistake in previous Tool Call target lines?
             # Target lines 105-147 is indeed inside `patched_step`.
             # `patched_scale_model_input` is lines 175+.
             
             # So my previous edit (Step 423) put the modification logic in `patched_step`? 
             # No, 105-147 is `patched_step`.
             # But `patched_step` is called AFTER UNet. Modifying sigma there is TOO LATE for UNet input scaling, 
             # but `step` uses sigma for sampling.
             # However, `scale_model_input` uses sigma for input scaling.
             # If we want consistent sigma, we must modify it BEFORE `scale_model_input`.
             
             # So `patched_scale_model_input` is the right place.
             # I should undo the change to `patched_step` (if I applied it) or ensure it's empty pass.
             # And put the logic in `patched_scale_model_input`.
             
             # Let's check what I did in Step 423.
             # I replaced lines 105-147 with the logic.
             # Lines 105-147 are inside `patched_step` in `__enter__`.
             # This means I moved logic to `patched_step`?
             # Wait, `patched_step` is mostly comments in original file.
             # If I enable it there, it runs AFTER `original_step`? No, I put it before `self.original_step`.
             
             # BUT `scale_model_input` runs BEFORE `step`.
             # So `patched_step` runs AFTER `scale_model_input` and UNet.
             # So logic in `patched_step` (before calling original step) affects `step` calculation but NOT UNet input scaling.
             # This might cause mismatch (UNet sees Sigma A, Sampler sees Sigma B).
             # Detail Daemon usually affects both?
             # "Detail Daemon modifies params.sigma ...". params.sigma is used for everything in A1111 loop.
             
             # So we must modify it EARLY.
             # `patched_scale_model_input` is early.
             # So logic SHOULD be in `patched_scale_model_input`.
             
             pass
             
    # I will replace the messy debug code in `patched_scale_model_input` with the CLEAN logic.
    # And I will revert `patched_step` to be simple.
             
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
                 
                 # Restore logic
                 self.saved_sigma_idx = idx
                 self.saved_sigma_val = original_sigma
                 
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
