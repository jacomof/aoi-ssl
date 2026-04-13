from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class iBotLoss(nn.Module):
    def __init__(
        self,
        student_temp_cls: float = 0.1,
        teacher_temp_cls: float = 0.04,
        center_momentum_cls: float = 0.9,
        student_temp_patch: float = 0.1,
        teacher_temp_patch: float = 0.04,
        center_momentum_patch: float = 0.9,
        out_dim: int = 256,
        **kwargs,
    ):
        super().__init__()
        self.student_temp_cls = float(student_temp_cls)
        self.teacher_temp_cls = float(teacher_temp_cls)
        self.center_momentum_cls = float(center_momentum_cls)
        self.student_temp_patch = float(student_temp_patch)
        self.teacher_temp_patch = float(teacher_temp_patch)
        self.center_momentum_patch = float(center_momentum_patch)
        self.register_buffer("center_cls", torch.zeros(1, int(out_dim)))
        self.register_buffer("center_patch", torch.zeros(1, 1, int(out_dim)))

    @torch.no_grad()
    def _update_center_cls(self, teacher_cls: torch.Tensor) -> None:
        batch_center = teacher_cls.mean(dim=0, keepdim=True)
        self.center_cls.mul_(self.center_momentum_cls).add_(
            (1.0 - self.center_momentum_cls) * batch_center
        )

    @torch.no_grad()
    def _update_center_patch(self, teacher_patch: torch.Tensor) -> None:
        batch_center = teacher_patch.mean(dim=(0, 1), keepdim=True)
        self.center_patch.mul_(self.center_momentum_patch).add_(
            (1.0 - self.center_momentum_patch) * batch_center
        )

    def softmax_center_teacher_cls(self, teacher_cls: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self._update_center_cls(teacher_cls)
        return F.softmax(
            (teacher_cls - self.center_cls) / self.teacher_temp_cls, dim=-1
        )

    def softmax_center_teacher_patch(self, teacher_patch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self._update_center_patch(teacher_patch)
        return F.softmax(
            (teacher_patch - self.center_patch) / self.teacher_temp_patch,
            dim=-1,
        )

    def dino_loss(
        self,
        student_cls_list: list[torch.Tensor] | tuple[torch.Tensor, ...],
        teacher_cls_list: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        total_loss = 0.0
        total_terms = 0
        for student_cls in student_cls_list:
            lsm = F.log_softmax(student_cls / self.student_temp_cls, dim=-1)
            for teacher_cls in teacher_cls_list:
                total_loss = total_loss + torch.sum(-teacher_cls * lsm, dim=-1).mean()
                total_terms += 1

        if total_terms == 0:
            return torch.tensor(0.0, device=self.center_cls.device)
        return total_loss / total_terms

    def mim_loss(
        self,
        student_patch_1: torch.Tensor,
        student_patch_2: torch.Tensor,
        teacher_patch_1: torch.Tensor,
        teacher_patch_2: torch.Tensor,
        masks,
    ) -> torch.Tensor:
        s1 = F.log_softmax(student_patch_1 / self.student_temp_patch, dim=-1)
        s2 = F.log_softmax(student_patch_2 / self.student_temp_patch, dim=-1)
        loss_1 = torch.sum(-teacher_patch_1 * s1, dim=-1).mean()
        loss_2 = torch.sum(-teacher_patch_2 * s2, dim=-1).mean()
        return 0.5 * (loss_1 + loss_2)
