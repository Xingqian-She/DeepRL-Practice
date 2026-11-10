"""配置集中在这里；模型结构与实验设置会一起写入 checkpoint。"""

from dataclasses import dataclass, field

from rl_shared import TransformerConfig

from sensors import SensorConfig


ENV_ID = "LunarLander-v3"


@dataclass(frozen=True)
class AgentConfig:
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    context_length: int = 32
    alpha_init: float = 1e-4
    use_previous_action: bool = True
    backbone_mode: str = "shared"

    def backbone_config(self):
        return TransformerConfig(
            hidden_dim=self.hidden_dim, num_layers=self.num_layers,
            num_heads=self.num_heads, context_length=self.context_length,
            alpha_init=self.alpha_init,
        )

    def __post_init__(self):
        self.backbone_config()  # 复用共享组件的结构校验，不改它的公开接口。
        if type(self.use_previous_action) is not bool:
            raise ValueError("use_previous_action 必须是 bool")
        if self.backbone_mode not in ("shared", "separate"):
            raise ValueError("backbone_mode 只能是 shared 或 separate")

    @property
    def warmup_length(self):
        # 多层注意力的感受野比单层窗口更长；额外保留一个 token 作为保守预热。
        return 1 + self.num_layers * (self.context_length - 1)


@dataclass(frozen=True)
class TrainConfig:
    agent: AgentConfig = field(default_factory=AgentConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    total_timesteps: int = 5_000_000
    num_envs: int = 8
    num_steps: int = 256
    sequence_length: int = 64
    sequences_per_minibatch: int = 4
    update_epochs: int = 4
    learning_rate: float = 3e-4
    lr_schedule: str = "constant"
    learning_rate_final: float = 0.0
    normalize_advantage: bool = True
    value_clip_coef: float = 0.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    ent_coef_schedule: str = "constant"
    ent_coef_final: float = 0.0
    max_grad_norm: float = 0.5
    consistency_tolerance: float = 5e-4
    consistency_warning_tolerance: float = 1e-4
    target_kl: float = 0.02
    eval_interval: int = 100  # PPO updates; 0 disables evaluation.
    eval_episodes: int = 20
    eval_num_envs: int = 8
    eval_seed: int = 10000
    eval_policy: str = "sample"
    curriculum_step: float = 0.1
    curriculum_return_threshold: float = 200.0
    curriculum_success_rate: float = 0.8
    curriculum_patience: int = 2
    max_episode_steps: int = 1000
    seed: int = 42
    device: str = "auto"
    compile_mode: str = "auto"
    save_interval: int = 10
    diagnostic_interval: int = 50

    def __post_init__(self):
        import math

        for name in ("total_timesteps", "num_envs", "num_steps", "sequence_length",
                     "sequences_per_minibatch", "update_epochs", "max_episode_steps", "save_interval",
                     "eval_episodes", "eval_num_envs", "curriculum_patience"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} 必须是正整数")
        if self.total_timesteps < self.num_envs * self.num_steps:
            raise ValueError("total_timesteps 至少需要容纳一个完整 rollout（num_envs × num_steps）")
        for name in ("learning_rate", "clip_coef", "max_grad_norm", "consistency_tolerance",
                     "consistency_warning_tolerance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须是有限正数")
        for name in ("gamma", "gae_lambda"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} 必须在 [0, 1] 内")
        for name in ("vf_coef", "ent_coef", "learning_rate_final", "ent_coef_final", "value_clip_coef"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} 必须是有限非负数")
        for name in ("lr_schedule", "ent_coef_schedule"):
            if getattr(self, name) not in ("constant", "linear"):
                raise ValueError(f"{name} 只能是 constant 或 linear")
        if type(self.normalize_advantage) is not bool:
            raise ValueError("normalize_advantage 必须是 bool")
        if type(self.diagnostic_interval) is not int or self.diagnostic_interval < 0:
            raise ValueError("diagnostic_interval 必须是非负整数；0 关闭诊断")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed 必须在 [0, 2**32) 内")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device 只能是 auto、cpu 或 cuda")
        if self.compile_mode not in ("auto", "on", "off"):
            raise ValueError("compile_mode 只能是 auto、on 或 off")
        if self.consistency_warning_tolerance > self.consistency_tolerance:
            raise ValueError("consistency_warning_tolerance 不能大于 consistency_tolerance")
        if not math.isfinite(self.target_kl) or self.target_kl < 0:
            raise ValueError("target_kl 必须是有限非负数；0 关闭提前停止")
        if type(self.eval_interval) is not int or self.eval_interval < 0:
            raise ValueError("eval_interval 必须是非负整数；0 关闭评估")
        if type(self.eval_seed) is not int or not 0 <= self.eval_seed < 2**32:
            raise ValueError("eval_seed 必须在 [0, 2**32) 内")
        if self.eval_policy not in ("argmax", "sample"):
            raise ValueError("eval_policy 只能是 argmax 或 sample")
        if not math.isfinite(self.curriculum_step) or not 0 < self.curriculum_step <= 1:
            raise ValueError("curriculum_step 必须在 (0, 1] 内")
        if not math.isfinite(self.curriculum_return_threshold):
            raise ValueError("curriculum_return_threshold 必须是有限数")
        if not math.isfinite(self.curriculum_success_rate) or not 0 <= self.curriculum_success_rate <= 1:
            raise ValueError("curriculum_success_rate 必须在 [0, 1] 内")
        if self.sensors.schedule == "performance" and (self.sensors.mode != "curriculum" or not self.eval_interval):
            raise ValueError("performance 课程需要 sensor-mode curriculum 且 eval-interval > 0")

    @property
    def num_updates(self):
        return self.total_timesteps // (self.num_envs * self.num_steps)

    @property
    def effective_total_timesteps(self):
        return self.num_updates * self.num_envs * self.num_steps
