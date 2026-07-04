import torch

from kernels.benchmark import Benchmark


class LionBenchmark(Benchmark):
    seed = 0

    def setup(self):
        self.p0 = torch.randn(2**24, dtype=torch.float32, device=self.device)
        self.m0 = torch.randn_like(self.p0)
        self.g = torch.randn_like(self.p0)
        self.p = self.p0.clone()
        self.m = self.m0.clone()
        self.out = self.p

    def benchmark_step(self):
        self.kernel.lion_step(self.p, self.m, self.g, 3e-4, 0.9, 0.99, 0.1)

    def verify_step(self) -> torch.Tensor:
        update = 0.9 * self.m0 + 0.1 * self.g
        return self.p0 - 3e-4 * torch.sign(update) - 3e-4 * 0.1 * self.p0
