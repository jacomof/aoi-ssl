import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class DINOLoss(nn.Module):
    def __init__(
        self,
        student_temp=0.1,  # Used in the paper
        center_momentum=0.9,  # best results in paper using 0.9, but  only collapse observed with 0.1
        teacher_temp=0.04,  # best results in paper using linear warmup from 0.04 to 0.07 during first 30 epochs
        rate=0.1,
        out_dim=256,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.updated = True
        self.reduce_handle = None
        self.teacher_temp = teacher_temp
        # self.len_teacher_output = None
        # self.async_batch_center = None

    def apply_center_update(self, teacher_output: torch.Tensor):
        batch_center = teacher_output.sum(dim=0, keepdim=True)
        batch_size = teacher_output.shape[0]

        if dist.is_initialized():
            # Synchronize self.center across all GPUs by averaging
            dist.all_reduce(batch_center, op=dist.ReduceOp.SUM)
            batch_center /= dist.get_world_size() * batch_size
        else:
            batch_center /= batch_size

        self.center = (
            self.center * self.center_momentum
            + (1 - self.center_momentum) * batch_center
        )

    def softmax_center_teacher(self, teacher_output):
        with torch.no_grad():
            self.apply_center_update(teacher_output)
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)

    def softmax_sharpen_only_teacher(self, teacher_output):
        # teacher centering and sharpening
        return F.softmax(teacher_output / self.teacher_temp, dim=-1)

    def softmax_center_only_teacher(self, teacher_output):
        with torch.no_grad():
            self.apply_center_update(teacher_output)
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center), dim=-1)

    def softmax_vanilla_teacher(self, teacher_output):
        # teacher centering and sharpening
        return F.softmax(teacher_output, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(
            teacher_output / teacher_temp
        ).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, student_output_list, teacher_out_softmaxed_centered_list):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        # Loss is computed by comparing the student
        # and teacher representations of each crop separately,
        # and then summing the losses.
        # This means that we are teaching the student to produce
        # the same representation of the teacher at the crop level.

        # It seems dino skips comparisons between the representations of the same view
        # and only compares the representations of different views.
        # However, DinoV2 does compare the representations of the same view.
        # teacher = [BxD, BxD]
        # student = [BxD, BxD, ..., BxD] -> nlocal_crops + 2
        total_loss = 0
        total_loss_elements = 0
        for s in student_output_list:
            lsm = F.log_softmax(s / self.student_temp, dim=-1)
            if torch.isnan(lsm).any():
                print("Log softmax of student outputs is NaN")
            for t in teacher_out_softmaxed_centered_list:
                if torch.any(t < 0):
                    print("Some teacher outputs are negative")
                if torch.any(lsm > 0):
                    print("Some student outputs LSM outputs are positive")
                if torch.any(t.isnan()):
                    print("Some teacher outputs are NaN")
                loss = torch.sum(
                    -t * lsm, dim=-1
                )  # Input: [batch_size, hidden_dim] Output: [batch_size]
                if loss.isnan().any():
                    print("Loss is NaN")
                total_loss += loss.mean()  # Input: [batch_size] Output: scalar
                total_loss_elements += 1

        return total_loss / total_loss_elements
