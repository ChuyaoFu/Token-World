# Adapted for the TokenWorld RoboTwin release: local imports and portable entrypoints.
# Source attribution and original licenses are listed in THIRD_PARTY_NOTICES.md.
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist

class FidelityMetrics:
    def __init__(self, feature_dim: int, latent_dim: int, device: torch.device) -> None:
        self.device = device
        # feature SSE, feature energy, cosine sum, cosine rows, values,
        # feature rows, pixel L1, pixel SSE, pixel values, latent rows,
        # posterior KL sum, posterior views, posterior values, posterior std sum
        self.scalars = torch.zeros(14, dtype=torch.float64, device=device)
        self.teacher_sum = torch.zeros(feature_dim, dtype=torch.float64, device=device)
        self.teacher_square_sum = torch.zeros(feature_dim, dtype=torch.float64, device=device)
        self.latent_sum = torch.zeros(latent_dim, dtype=torch.float64, device=device)
        self.latent_outer = torch.zeros(latent_dim, latent_dim, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(
        self,
        teacher: torch.Tensor,
        reconstruction: torch.Tensor,
        latents: torch.Tensor,
        pixel_target: torch.Tensor | None = None,
        pixel_prediction: torch.Tensor | None = None,
        posterior_mean: torch.Tensor | None = None,
        posterior_logvar: torch.Tensor | None = None,
    ) -> None:
        teacher = teacher.double()
        reconstruction = reconstruction.double()
        error = reconstruction - teacher
        flat_teacher = teacher.reshape(-1, teacher.shape[-1])
        flat_reconstruction = reconstruction.reshape(-1, reconstruction.shape[-1])
        self.scalars[0] += error.square().sum()
        self.scalars[1] += teacher.square().sum()
        cosine = F.cosine_similarity(flat_reconstruction, flat_teacher, dim=-1)
        self.scalars[2] += cosine.sum()
        self.scalars[3] += cosine.numel()
        self.scalars[4] += teacher.numel()
        self.scalars[5] += flat_teacher.shape[0]
        self.teacher_sum += flat_teacher.sum(dim=0)
        self.teacher_square_sum += flat_teacher.square().sum(dim=0)

        flat_latents = latents.double().reshape(-1, latents.shape[-1])
        self.latent_sum += flat_latents.sum(dim=0)
        self.latent_outer += flat_latents.T @ flat_latents
        self.scalars[9] += flat_latents.shape[0]

        if pixel_target is not None and pixel_prediction is not None:
            target = pixel_target.double()
            prediction = pixel_prediction.double()
            pixel_error = prediction - target
            self.scalars[6] += pixel_error.abs().sum()
            self.scalars[7] += pixel_error.square().sum()
            self.scalars[8] += pixel_error.numel()

        if posterior_mean is not None and posterior_logvar is not None:
            mean = posterior_mean.double()
            logvar = posterior_logvar.double()
            kl_elements = -0.5 * (1.0 + logvar - mean.square() - logvar.exp())
            self.scalars[10] += kl_elements.sum()
            self.scalars[11] += mean.shape[0]
            self.scalars[12] += mean.numel()
            self.scalars[13] += torch.exp(0.5 * logvar).sum()

    def reduce(self) -> None:
        if dist.is_initialized():
            for tensor in (self.scalars, self.teacher_sum, self.teacher_square_sum, self.latent_sum, self.latent_outer):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def compute(self) -> dict:
        self.reduce()
        values = self.scalars.cpu()
        sse = float(values[0])
        energy = float(values[1])
        feature_rows = max(float(values[5]), 1.0)
        centered_energy = float(
            (
                self.teacher_square_sum
                - self.teacher_sum.square() / feature_rows
            ).sum().cpu()
        )
        latent_rows = max(float(values[9]), 1.0)
        latent_mean = self.latent_sum / latent_rows
        covariance = self.latent_outer / latent_rows - torch.outer(latent_mean, latent_mean)
        eigenvalues = torch.linalg.eigvalsh(covariance.cpu()).clamp_min(0.0)
        eigen_sum = float(eigenvalues.sum())
        if eigen_sum > 0:
            probabilities = eigenvalues / eigen_sum
            positive = probabilities > 0
            effective_rank = float(torch.exp(-(probabilities[positive] * probabilities[positive].log()).sum()))
        else:
            effective_rank = 0.0
        result = {
            "feature_mse": sse / max(float(values[4]), 1.0),
            "feature_nmse": sse / max(energy, 1.0e-12),
            "feature_relative_rmse": math.sqrt(sse / max(energy, 1.0e-12)),
            "feature_cosine": float(values[2]) / max(float(values[3]), 1.0),
            "feature_r2": 1.0 - sse / max(centered_energy, 1.0e-12),
            "feature_rows": int(values[5]),
            "latent_mean": float(latent_mean.mean().cpu()),
            "latent_std_mean": float(torch.sqrt(torch.diag(covariance).clamp_min(0.0)).mean().cpu()),
            "latent_effective_rank": effective_rank,
            "latent_covariance_condition": (
                float(eigenvalues.max() / eigenvalues[eigenvalues > 1.0e-12].min())
                if bool((eigenvalues > 1.0e-12).any())
                else None
            ),
        }
        if float(values[8]) > 0:
            pixel_mse = float(values[7]) / float(values[8])
            result.update(
                {
                    "pixel_l1": float(values[6]) / float(values[8]),
                    "pixel_mse": pixel_mse,
                    "pixel_psnr": -10.0 * math.log10(max(pixel_mse, 1.0e-12)),
                }
            )
        if float(values[12]) > 0:
            result.update(
                {
                    "posterior_kl_sum_per_view": float(values[10]) / float(values[11]),
                    "posterior_kl_per_dimension": float(values[10]) / float(values[12]),
                    "posterior_std_mean": float(values[13]) / float(values[12]),
                }
            )
        return result

