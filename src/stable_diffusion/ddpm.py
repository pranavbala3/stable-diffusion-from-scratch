import torch
import numpy as np

class DDPMSampler:

    def __init__(self, generator: torch.Generator, training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        # beta_start: the starting value for the variance schedule
        # beta_end: the ending value for the variance schedule
        # same values used in the DDPM paper
        
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, training_steps, dtype=torch.float32) ** 2  # scaled linear schedule

        # alphas to get to any time step
        self.alphas = 1.0 - self.betas
        
        self.cumprod_alphas = torch.cumprod(self.alphas, 0) # [a_0, a_0 * a_1, a_0 * a_1 * a_2, ...]
        self.one = torch.tensor(1.0)
        self.generator = generator
        self.training_steps = training_steps
        self.timesteps = torch.from_numpy(np.arange(0, training_steps)[::-1].copy()) # timesteps for inference process (1000 -> 0)

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - self.training_steps // self.inference_steps
        return prev_t

    def _get_variance(self, timestep: int) -> torch.Tensor:
        prev_t = self._get_previous_timestep(timestep)
        alpha_prod_t = self.cumprod_alphas[timestep]
        alpha_prod_t_prev = self.cumprod_alphas[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # for t > 0, compute predicted variance beta_t from formulas 6 and 7 in DDPM paper and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # clamp variance to ensure it is not 0
        variance = torch.clamp(variance, min=1e-20)
        return variance

    # space out the timesteps for inference
    # ex. if training_steps = 1000, inference_steps = 50, then timesteps = [1000, 990, 980, ..., 0]
    def set_inference_timesteps(self, inference_steps=50):
        self.inference_steps = inference_steps
        step_size = self.training_steps // self.inference_steps
        timesteps = (np.arange(0, inference_steps) * step_size).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    # set how much noise to add to the input image when doing image-to-image
    # more noise (more strength) means that the output will be further from the input image
    # less noise (less noise) means that the output will be closer to the input image
    def set_strength(self, strength=1):
        # start_step is the number of noise levels to skip
        start_step = self.inference_steps - int(self.inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # compute alphas, betas
        alpha_prod_t = self.cumprod_alphas[t]
        alpha_prod_t_prev = self.cumprod_alphas[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula 15 in DDPM paper
        pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        # compute coefficients for pred_original_sample x_0 and current sample x_t, formula 7 in DDPM paper
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # compute predicted previous sample mu_t, formula 7 in DDPM paper
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        # add noise
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            # Compute the variance, formula 7 in DDPM paper
            # sigma * N(0, 1)
            variance = (self._get_variance(t) ** 0.5) * noise
        
        # sample from N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # mu + variance [sigma * N(0, 1)]
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    # adding noise to the latent when doing image to image
    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        cumprod_alphas = self.cumprod_alphas.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = cumprod_alphas[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        # shape the tensor to match the original_samples shape
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # standard deviation of the distribution
        sqrt_one_minus_alpha_prod = (1 - cumprod_alphas[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # sample from q(x_t | x_0), formula 4 in DDPM paper
        # N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1) where mu = sqrt_alpha_prod * original_samples and sigma = sqrt_one_minus_alpha_prod
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples