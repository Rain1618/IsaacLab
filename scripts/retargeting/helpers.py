from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.managers import SceneEntityCfg

from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # type: ignore


def get_robot_cfg(robot_name: str) -> ArticulationCfg:
    """Simple validation method for checks"""
    if robot_name != "anymal_d":
        raise ValueError(f"Unsupported robot: {robot_name!r}")
    
    robot_cfg: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    return robot_cfg


ROBOT_META = {
    "anymal_d": {
        "foot_bodies": ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"],
        "leg_roots": ["LF_HIP", "RF_HIP", "LH_HIP", "RH_HIP"],
        "joint_names": [
            "LF_HAA", "LF_HFE", "LF_KFE",
            "RF_HAA", "RF_HFE", "RF_KFE",
            "LH_HAA", "LH_HFE", "LH_KFE",
            "RH_HAA", "RH_HFE", "RH_KFE",
        ],
        "default_joint_pos": {
            "LF_HAA": 0.0, "LF_HFE": 0.4, "LF_KFE": -0.8,
            "RF_HAA": 0.0, "RF_HFE": 0.4, "RF_KFE": -0.8,
            "LH_HAA": 0.0, "LH_HFE": -0.4, "LH_KFE": 0.8,
            "RH_HAA": 0.0, "RH_HFE": -0.4, "RH_KFE": 0.8,
        },
        # Per-suffix L2 weights on (q - q_default). Because every leg reuses
        # the same three values, this is symmetric across L/R and front/hind
        # by construction. HAA is penalised the hardest to keep hips close to
        # their neutral abduction; HFE/KFE are freer since they drive foot
        # reach. Override per-run via QuadrupedRetargeter(default_pose_W=...).
        "default_pose_W_per_suffix": {
            "HAA": 20.0,
            "HFE": 2.0,
            "KFE": 2.0,
        },
        "leg_length": 0.55,
        "base_height": 0.55,
        "stance_length": 0.70,
        "stance_width": 0.38,
    }
}


@dataclass
class ReferenceMotion:
    root_pos: np.ndarray
    root_rot: np.ndarray  # wxyz
    foot_pos: np.ndarray
    thigh_pos: np.ndarray
    root_lin_vel: np.ndarray | None
    root_ang_vel: np.ndarray | None
    dt: float


def _normalize_quaternions_np(quats: np.ndarray) -> np.ndarray:
    quats = np.asarray(quats, dtype=np.float32).copy()
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    quats /= np.clip(norms, 1e-8, None)
    return quats


def ensure_quaternion_continuity(quats: np.ndarray) -> np.ndarray:
    quats = _normalize_quaternions_np(quats)
    for t in range(1, quats.shape[0]):
        if np.dot(quats[t - 1], quats[t]) < 0.0:
            quats[t] *= -1.0
    return quats


def _quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    out = np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )
    return out.astype(np.float32)


def quat_conjugate_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).copy()
    q[..., 1:] *= -1.0
    return q


def quat_apply_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = _normalize_quaternions_np(q)
    v_as_quat = np.concatenate([np.zeros(v.shape[:-1] + (1,), dtype=np.float32), v.astype(np.float32)], axis=-1)
    rotated = _quat_mul_np(_quat_mul_np(q, v_as_quat), quat_conjugate_np(q))
    return rotated[..., 1:]


def world_to_local_points(root_pos: np.ndarray, root_rot: np.ndarray, pts_world: np.ndarray) -> np.ndarray:
    rel = pts_world - root_pos[..., None, :] if root_pos.ndim == 2 else pts_world - root_pos
    q_inv = quat_conjugate_np(root_rot)
    if rel.ndim == 3:
        q_inv = np.repeat(q_inv[:, None, :], rel.shape[1], axis=1)
    return quat_apply_np(q_inv, rel)


def local_to_world_points(root_pos: np.ndarray, root_rot: np.ndarray, pts_local: np.ndarray) -> np.ndarray:
    q = root_rot
    if pts_local.ndim == 3:
        q = np.repeat(q[:, None, :], pts_local.shape[1], axis=1)
    return quat_apply_np(q, pts_local) + (root_pos[..., None, :] if root_pos.ndim == 2 else root_pos)


def _quat_to_angular_velocity(quats: np.ndarray, dt: float) -> np.ndarray:
    quats = ensure_quaternion_continuity(quats)
    T = quats.shape[0]
    omega = np.zeros((T, 3), dtype=np.float32)
    q0 = quats[:-1]
    q1 = quats[1:]
    qr = _quat_mul_np(q1, quat_conjugate_np(q0))
    omega[1:] = 2.0 * qr[:, 1:] / dt
    if T > 1:
        omega[0] = omega[1]
    return omega

def load_reference_motion(path: str, fps: float) -> ReferenceMotion:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Reference motion not found: {path}")

    try:
        data = dict(np.load(path, allow_pickle=True))
    except Exception as exc:
        raise ValueError(f"Unsupported reference motion format: {p.suffix.lower()!r}") from exc

    def _get(key: str, required: bool = True) -> np.ndarray | None:
        val = data.get(key)
        if val is None and required:
            raise KeyError(
                f"Required key '{key}' missing from reference motion file. "
                f"Available keys: {list(data.keys())}"
            )
        return np.asarray(val, dtype=np.float32) if val is not None else None

    root_pos = _get("root_pos")
    root_rot = ensure_quaternion_continuity(_get("root_rot"))
    foot_pos = _get("foot_pos")
    thigh_pos = _get("thigh_pos")                      # NEW: required
    root_lin_vel = _get("root_lin_vel", required=False)
    root_ang_vel = _get("root_ang_vel", required=False)

    if root_pos is None or root_rot is None or foot_pos is None or thigh_pos is None:
        raise ValueError("Reference motion is missing required arrays.")

    T = root_pos.shape[0]
    assert root_rot.shape == (T, 4), f"root_rot must be (T,4), got {root_rot.shape}"
    assert foot_pos.shape == (T, 4, 3), f"foot_pos must be (T,4,3), got {foot_pos.shape}"
    assert thigh_pos.shape == (T, 4, 3), f"thigh_pos must be (T,4,3), got {thigh_pos.shape}"

    dt = 1.0 / fps
    if root_lin_vel is None:
        root_lin_vel = np.gradient(root_pos, dt, axis=0).astype(np.float32)
    if root_ang_vel is None:
        root_ang_vel = _quat_to_angular_velocity(root_rot, dt)

    print(f"[Loader] Loaded '{p.name}' — {T} frames @ {fps} Hz ({T / fps:.1f} s).")
    return ReferenceMotion(
        root_pos=root_pos,
        root_rot=root_rot,
        foot_pos=foot_pos,
        thigh_pos=thigh_pos,
        root_lin_vel=root_lin_vel,
        root_ang_vel=root_ang_vel,
        dt=dt,
    )

def compute_scale_factors(ref_motion: ReferenceMotion, meta: dict) -> dict:
    # Thigh positions in the animal's body frame (T, 4, 3)
    thigh_local = world_to_local_points(ref_motion.root_pos, ref_motion.root_rot, ref_motion.thigh_pos)
    # Mean over time gives a stable per-leg body-frame attachment (4, 3)
    thigh_local_mean = thigh_local.mean(axis=0)

    # Lateral: |y| gap between left (idx 0: LF) and right (idx 1: RF) thighs.
    # Longitudinal: |x| gap between front (idx 0: LF) and hind (idx 2: LH) thighs.
    lat_animal = float(np.abs(thigh_local_mean[0, 1] - thigh_local_mean[1, 1]))
    lon_animal = float(np.abs(thigh_local_mean[0, 0] - thigh_local_mean[2, 0]))

    # z still derived from body clearance above the feet.
    mean_foot_h = ref_motion.foot_pos[:, :, 2].mean(axis=1)
    base_h_animal = float(np.median(ref_motion.root_pos[:, 2] - mean_foot_h))

    robot_width = float(meta["stance_width"])
    robot_length = float(meta["stance_length"])
    robot_base_h = float(meta["base_height"])

    sx = robot_length / max(lon_animal, 1e-4)
    sy = robot_width / max(lat_animal, 1e-4)
    sz = robot_base_h / max(base_h_animal, 1e-4)

    scale = {
        "x": float(sx),
        "y": float(sy),
        "z": float(sz),
        "base_h_animal": float(base_h_animal),
        "robot_base_h": robot_base_h,
        "animal_stance_length": float(lon_animal),
        "animal_stance_width": float(lat_animal),
        "robot_stance_length": robot_length,
        "robot_stance_width": robot_width,
    }
    print(
        "[Scale] (from thighs) "
        f"animal length≈{lon_animal:.3f}m width≈{lat_animal:.3f}m base_h≈{base_h_animal:.3f}m | "
        f"robot length≈{robot_length:.3f}m width≈{robot_width:.3f}m base_h≈{robot_base_h:.3f}m | "
        f"scale xyz=({sx:.3f}, {sy:.3f}, {sz:.3f})"
    )
    return scale


class QuadrupedRetargeter:
    def __init__(
        self,
        robot: Articulation,
        scene: InteractiveScene,
        sim,
        meta: dict,
        device: str,
        ik_iterations: int,
        ik_damping: float,
        physics_dt: float,
        num_envs: int,
        default_pose_W: np.ndarray | dict | None = None,  # NEW
    ):
        self.robot = robot
        self.scene = scene
        self.sim = sim
        self.meta = meta
        self.device = device
        self.ik_iters = ik_iterations
        self.physics_dt = physics_dt
        self.num_envs = num_envs
        self.is_fixed_base = bool(getattr(robot, "is_fixed_base", False))

        ik_cfg = DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": ik_damping},
        )
        self.ik_controllers = [
            DifferentialIKController(ik_cfg, num_envs=num_envs, device=device) for _ in range(4)
        ]

        all_joint_names = list(robot.joint_names)
        all_body_names = list(robot.body_names)
        self.leg_joint_ids: list[list[int]] = []
        for leg_i in range(4):
            leg_names = meta["joint_names"][leg_i * 3: leg_i * 3 + 3]
            self.leg_joint_ids.append([all_joint_names.index(name) for name in leg_names])

        self.foot_body_ids  = [robot.body_names.index(n) for n in meta["foot_bodies"]]
        self.thigh_body_ids = [robot.body_names.index(n) for n in meta["leg_roots"]]

        # Isaac Lab's Jacobian body axis EXCLUDES the root link on floating-base
        # articulations too; the "-1 only if fixed_base" logic in solve_frame is
        # incorrect for ANYmal-D. Precompute the Jacobian-axis indices here.
        root_link_idx = 0  # ANYmal-D's root body is at index 0
        self.foot_jac_idx  = [i - 1 if i > root_link_idx else i for i in self.foot_body_ids]
        self.thigh_jac_idx = [i - 1 if i > root_link_idx else i for i in self.thigh_body_ids]

        print("[IK] body-name resolution:")
        for leg_i, name in enumerate(meta["foot_bodies"]):
            print(f"  leg {leg_i} {name:8s}: body_id={self.foot_body_ids[leg_i]:2d}  jac_idx={self.foot_jac_idx[leg_i]:2d}")
        for leg_i, name in enumerate(meta["leg_roots"]):
            print(f"  leg {leg_i} {name:8s}: body_id={self.thigh_body_ids[leg_i]:2d}  jac_idx={self.thigh_jac_idx[leg_i]:2d}")

        default_map: dict = meta["default_joint_pos"]

        # self note: qpos means generalized position vector
        # naming scheme from mujoco
        self.default_qpos = self.robot.data.default_joint_pos.clone()
        for joint_name, value in default_map.items():
            joint_id = all_joint_names.index(joint_name)
            self.default_qpos[:, joint_id] = float(value)

        # Fixed base pose for "pinned in the air" retargeting.
        # Robot root is locked at (0, 0, 0.5) world with identity orientation (wxyz).
        self.fixed_root_pos = torch.tensor(
            [[0.0, 0.0, 0.5]], dtype=torch.float32, device=device
        ).expand(num_envs, -1).contiguous()
        self.fixed_root_rot = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device
        ).expand(num_envs, -1).contiguous()

        self.ik_damping = float(ik_damping)  # keep for the augmented DLS

        # ---- Default-pose L2 regulariser -----------------------------------
        # W is a (num_dofs,) non-negative diagonal. We only ever use W^T W,
        # so we store w_sq = W**2 directly. `default_pose_W` can be:
        #   - None                 -> use meta["default_pose_W_per_suffix"].
        #   - dict {suffix: w}     -> per-suffix scalar, broadcast across legs
        #                             (L/R symmetric by construction).
        #   - np.ndarray (12,)     -> explicit per-joint weights.
        w_vec = self._resolve_default_pose_weights(default_pose_W, meta, all_joint_names)
        self.default_pose_w_sq = torch.tensor(
            (w_vec ** 2).astype(np.float32), device=device
        )  # shape (num_dofs,)

        # Per-leg views (HAA, HFE, KFE) for use inside solve_frame.
        self.leg_w_sq: list[torch.Tensor] = []
        for leg_i in range(4):
            leg_ids = self.leg_joint_ids[leg_i]
            self.leg_w_sq.append(self.default_pose_w_sq[torch.tensor(leg_ids, device=device)])

        print(
            "[IK] default-pose L2 weights (diag W) per joint:\n  "
            + ", ".join(f"{n}={w:.2f}" for n, w in zip(all_joint_names, w_vec))
        )

    @staticmethod
    def _resolve_default_pose_weights(
        default_pose_W, meta: dict, all_joint_names: list[str]
    ) -> np.ndarray:
        """Expand a user-provided weight spec into a (num_dofs,) vector
        aligned with `all_joint_names`. Enforces L/R symmetry when a
        per-suffix dict is used."""
        num_dofs = len(all_joint_names)

        if default_pose_W is None:
            suffix_map = meta.get("default_pose_W_per_suffix", {"HAA": 1.0, "HFE": 1.0, "KFE": 1.0})
            w = np.zeros(num_dofs, dtype=np.float32)
            for i, name in enumerate(all_joint_names):
                suffix = name.split("_")[-1]
                w[i] = float(suffix_map.get(suffix, 1.0))
            return w

        if isinstance(default_pose_W, dict):
            w = np.zeros(num_dofs, dtype=np.float32)
            for i, name in enumerate(all_joint_names):
                if name in default_pose_W:                 # exact joint name wins
                    w[i] = float(default_pose_W[name])
                else:
                    suffix = name.split("_")[-1]
                    w[i] = float(default_pose_W.get(suffix, 0.0))
            return w

        w = np.asarray(default_pose_W, dtype=np.float32).reshape(-1)
        if w.shape[0] != num_dofs:
            raise ValueError(
                f"default_pose_W array must have length {num_dofs} "
                f"(one per joint); got {w.shape[0]}."
            )
        if np.any(w < 0.0):
            raise ValueError("default_pose_W must be non-negative.")
        return w

    def _flush_state(self, joint_pos, joint_vel=None):
        if joint_vel is None:
            joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)        # physics-only during inner IK
        self.scene.update(self.physics_dt)
        self.robot.update(self.physics_dt)

        self.sim.render()

    def _teleport_base(self, root_pos_t: torch.Tensor, root_rot_t: torch.Tensor) -> None:
        root_state = self.robot.data.default_root_state.clone()
        root_state[:, :3] = root_pos_t
        root_state[:, 3:7] = root_rot_t
        root_state[:, 7:] = 0.0
        self.robot.write_root_pose_to_sim(root_state[:, :7])
        self.robot.write_root_velocity_to_sim(root_state[:, 7:])

    def solve_frame(
        self,
        root_pos: np.ndarray,
        root_rot: np.ndarray,
        foot_pos_local: np.ndarray,
        thigh_pos_local: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        dev = self.device
        E = self.num_envs
        lam2 = self.ik_damping ** 2

        # Base is pinned.
        fixed_rp = self.fixed_root_pos
        fixed_rr = self.fixed_root_rot

        # helpers.py: solve_frame — fix rank-2 expansion
        rp = torch.tensor(root_pos, dtype=torch.float32, device=dev).unsqueeze(0).expand(E, -1)
        rr = torch.tensor(root_rot, dtype=torch.float32, device=dev).unsqueeze(0).expand(E, -1)

        foot_local = torch.tensor(foot_pos_local, dtype=torch.float32, device=dev).unsqueeze(0).expand(E, -1, -1)
        thigh_local = torch.tensor(thigh_pos_local, dtype=torch.float32, device=dev).unsqueeze(0).expand(E, -1, -1)

        self._teleport_base(rp, rr)

        jpos = self.default_qpos.clone()
        jvel = torch.zeros_like(jpos)
        self._flush_state(jpos, jvel)

        # Default joint positions as a reusable tensor view.
        q_def = self.default_qpos  # (E, num_dofs)

        for _ in range(self.ik_iters):
            jacobian_full = self.robot.root_physx_view.get_jacobians()

            for leg_i in range(4):
                leg_ids = self.leg_joint_ids[leg_i]           # [HAA, HFE, KFE]
                haa_id = leg_ids[0]
                hfe_kfe_ids = leg_ids[1:]
                foot_id = self.foot_body_ids[leg_i]
                thigh_id = self.thigh_body_ids[leg_i]
                foot_body_idx  = self.foot_jac_idx[leg_i]
                thigh_body_idx = self.thigh_jac_idx[leg_i]

                leg_w_sq = self.leg_w_sq[leg_i]               # (3,) ordered as leg_ids

                # ---- (1) THIGH TARGET: HAA-only augmented DLS --------------
                thigh_pos_w = self.robot.data.body_pos_w[:, thigh_id, :]
                thigh_quat_w = self.robot.data.body_quat_w[:, thigh_id, :]
                thigh_pos_b, _ = subtract_frame_transforms(rp, rr, thigh_pos_w, thigh_quat_w)

                J_haa = jacobian_full[:, thigh_body_idx, :3, :][:, :, [haa_id]]    # (E, 3, 1)
                err_thigh = (thigh_local[:, leg_i, :] - thigh_pos_b).unsqueeze(-1)  # (E, 3, 1)

                w_sq_haa = leg_w_sq[0]                         # scalar weight^2 for HAA
                q_dev_haa = (jpos[:, [haa_id]] - q_def[:, [haa_id]]).unsqueeze(-1)  # (E, 1, 1)

                # A = J^T J + (lambda^2 + w^2) I,   size (E, 1, 1)
                A = torch.matmul(J_haa.transpose(1, 2), J_haa) + (lam2 + w_sq_haa) * torch.eye(1, device=dev).unsqueeze(0)
                # b = J^T e - w^2 (q - q_def)
                b = torch.matmul(J_haa.transpose(1, 2), err_thigh) - w_sq_haa * q_dev_haa
                dq_haa = torch.linalg.solve(A, b).squeeze(-1)                         # (E, 1)
                jpos[:, haa_id] = jpos[:, haa_id] + dq_haa[:, 0]

                lo_haa = self.robot.data.soft_joint_pos_limits[:, haa_id, 0]
                hi_haa = self.robot.data.soft_joint_pos_limits[:, haa_id, 1]
                jpos[:, haa_id] = torch.clamp(jpos[:, haa_id], lo_haa, hi_haa)

                # ---- (2) FOOT TARGET: HFE+KFE augmented DLS ---------------
                ee_pos_w = self.robot.data.body_pos_w[:, foot_id, :]
                ee_quat_w = self.robot.data.body_quat_w[:, foot_id, :]
                ee_pos_b, _ = subtract_frame_transforms(rp, rr, ee_pos_w, ee_quat_w)

                target_pos_b = foot_local[:, leg_i, :]
                err_foot = (target_pos_b - ee_pos_b).unsqueeze(-1)                     # (E, 3, 1)

                hfe_kfe_dof_ids = torch.tensor(hfe_kfe_ids, device=dev, dtype=torch.long)
                J_foot = jacobian_full[:, foot_body_idx, :3, :][:, :, hfe_kfe_dof_ids]  # (E, 3, 2)

                W_sub = leg_w_sq[1:]                                                   # (2,) HFE, KFE
                W_diag = torch.diag(W_sub).unsqueeze(0).expand(E, -1, -1)              # (E, 2, 2)
                q_dev_sub = (jpos[:, hfe_kfe_ids] - q_def[:, hfe_kfe_ids]).unsqueeze(-1)  # (E, 2, 1)

                I2 = torch.eye(2, device=dev).unsqueeze(0)
                A = torch.matmul(J_foot.transpose(1, 2), J_foot) + lam2 * I2 + W_diag  # (E, 2, 2)
                b = torch.matmul(J_foot.transpose(1, 2), err_foot) - torch.matmul(W_diag, q_dev_sub)
                dq_sub = torch.linalg.solve(A, b).squeeze(-1)                          # (E, 2)
                jpos[:, hfe_kfe_ids] = jpos[:, hfe_kfe_ids] + dq_sub

                lo = self.robot.data.soft_joint_pos_limits[:, hfe_kfe_ids, 0]
                hi = self.robot.data.soft_joint_pos_limits[:, hfe_kfe_ids, 1]
                jpos[:, hfe_kfe_ids] = torch.clamp(jpos[:, hfe_kfe_ids], lo, hi)

            self._teleport_base(rp, rr)
            self._flush_state(jpos, jvel)

        self.robot.update(self.physics_dt)

        return (
            jpos[0].detach().cpu().numpy().astype(np.float32),
            self.robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32),
        )