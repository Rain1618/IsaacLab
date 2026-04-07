import torch

# Python default fields
from dataclasses import dataclass, field
 
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.controllers import (  # noqa: E402
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    quat_apply,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    subtract_frame_transforms,
)

import numpy as np


# Get the Anymal-D Articulation Configurations for performing IK
def _get_robot_cfg(robot_name: str) -> ArticulationCfg:
    """Return IsaacLab ArticulationCfg for the requested robot."""
    from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # type: ignore
    return ANYMAL_D_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@dataclass
class ReferenceMotion:
    """Container for raw animal reference motion data.
 
    Coordinate convention expected in the file
    -------------------------------------------
    root_pos  : (T, 3)   — XYZ position of the animal's pelvis/base [m]
    root_rot  : (T, 4)   — quaternion (w, x, y, z) of pelvis orientation
    foot_pos  : (T, 4, 3)— world-space XYZ of [FL, FR, RL, RR] feet [m]
    root_lin_vel : (T, 3)  — optional linear velocity in world frame [m/s]
    root_ang_vel : (T, 3)  — optional angular velocity in body frame [rad/s]
    dt        : float    — time between frames [s]
    """
    root_pos:     np.ndarray          # (T, 3)
    root_rot:     np.ndarray          # (T, 4)  wxyz
    foot_pos:     np.ndarray          # (T, 4, 3)
    root_lin_vel: np.ndarray | None   # (T, 3)  or None
    root_ang_vel: np.ndarray | None   # (T, 3)  or None
    dt:           float
    
    
# Below are some helper functions to for calculating missing data fields
def _quat_to_angular_velocity(quats: np.ndarray, dt: float) -> np.ndarray:
    """Numerical angular velocity from a quaternion sequence (wxyz convention)."""
    T = quats.shape[0]
    omega = np.zeros((T, 3), dtype=np.float32)
    for t in range(1, T):
        q0 = quats[t - 1]  # w,x,y,z
        q1 = quats[t]
        # q_rel = q1 * inv(q0)
        q0_conj = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
        qr = _quat_mul_np(q1, q0_conj)
        # omega ≈ 2 * qr.xyz / dt  (small-angle approximation)
        omega[t] = 2.0 * qr[1:] / dt
    omega[0] = omega[1]
    return omega
 
 
def _quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication (wxyz convention) — NumPy, scalar."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)
    
class QuadrupedRetargeter:
    """
    Drives an IsaacLab Articulation through a sequence of IK solves
    to retarget a reference motion.
 
    For each frame:
      1. Teleport the robot base to the (scaled) reference root pose.
      2. For each leg, set the foot EE target to the (scaled) reference
         foot position expressed in the robot's local base frame.
      3. Run the DifferentialIKController until convergence.
      4. Record joint positions, velocities, and full robot state.
    """
 
    def __init__(
        self,
        robot: Articulation,
        meta: dict,
        device: str,
        ik_iterations: int,
        ik_damping: float,
        physics_dt: float,
        num_envs: int,
    ):
        self.robot        = robot
        self.meta         = meta
        self.device       = device
        self.ik_iters     = ik_iterations
        self.physics_dt   = physics_dt
        self.num_envs     = num_envs
 
        n_legs = 4
 
        # ── Build one DifferentialIKController per leg ───────────────────────
        # Each controller operates on a 3-DoF chain: hip → thigh → calf/foot
        ik_cfg = DifferentialIKControllerCfg(
            command_type="position",   # we supply 3-D position targets
            use_relative_mode=False,
            ik_method="dls",           # damped-least-squares
            ik_params={"lambda_val": ik_damping},
        )
        self.ik_controllers: list[DifferentialIKController] = []
        for leg_i in range(n_legs):
            ctrl = DifferentialIKController(ik_cfg, num_envs=num_envs, device=device)
            self.ik_controllers.append(ctrl)
 
        # ── Joint index mapping per leg ──────────────────────────────────────
        # IsaacLab's Articulation.joint_names gives the full ordered list;
        # we build index slices for each 3-DoF leg.
        all_joint_names: list[str] = robot.joint_names
        self.leg_joint_ids: list[list[int]] = []
        for leg_i in range(n_legs):
            leg_jnames = meta["joint_names"][leg_i * 3: leg_i * 3 + 3]
            ids = [all_joint_names.index(jn) for jn in leg_jnames]
            self.leg_joint_ids.append(ids)
 
        # ── EE body indices ──────────────────────────────────────────────────
        all_body_names: list[str] = robot.body_names
        self.foot_body_ids = [
            all_body_names.index(bn) for bn in meta["foot_bodies"]
        ]
 
        # ── Default joint positions tensor ───────────────────────────────────
        self.default_qpos = torch.tensor(
            meta["default_joint_pos"], dtype=torch.float32, device=device
        ).unsqueeze(0).expand(num_envs, -1)    # (E, 12)
 
    # ─────────────────────────────────────────────────────────────────────────
 
    def _teleport_base(
        self,
        root_pos_t: torch.Tensor,   # (E, 3)
        root_rot_t: torch.Tensor,   # (E, 4) wxyz
    ):
        """Hard-set the robot root state (teleport, no physics step)."""
        root_state = self.robot.data.default_root_state.clone()  # (E, 13)
        root_state[:, :3]  = root_pos_t
        root_state[:, 3:7] = root_rot_t                          # wxyz → IsaacLab wxyz
        root_state[:, 7:]  = 0.0                                  # zero velocities
        self.robot.write_root_state_to_sim(root_state)
 
    # ─────────────────────────────────────────────────────────────────────────
 
    def solve_frame(
        self,
        root_pos: np.ndarray,   # (3,)
        root_rot: np.ndarray,   # (4,) wxyz
        foot_pos: np.ndarray,   # (4, 3)  world-space
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run IK for one motion frame.
 
        Returns
        -------
        joint_pos : np.ndarray  (12,)
        joint_vel : np.ndarray  (12,)  — estimated by finite-diff inside controller
        """
        dev = self.device
        E   = self.num_envs
        n_legs = 4
 
        # ── Convert reference frame data to tensors ───────────────────────────
        rp = torch.tensor(root_pos, dtype=torch.float32, device=dev).unsqueeze(0).expand(E, -1)
        rr = torch.tensor(root_rot, dtype=torch.float32, device=dev).unsqueeze(0).expand(E, -1)
        fp = torch.tensor(foot_pos, dtype=torch.float32, device=dev).unsqueeze(0).expand(E, -1, -1)
 
        # ── Teleport base, reset joint positions to default ───────────────────
        self._teleport_base(rp, rr)
 
        jpos = self.default_qpos.clone()   # (E, 12)
        jvel = torch.zeros_like(jpos)
 
        # Write initial joint positions
        joint_state = self.robot.data.default_joint_state.clone()
        joint_state[:, :, 0] = jpos          # positions
        joint_state[:, :, 1] = jvel          # velocities
        self.robot.write_joint_state_to_sim(
            joint_state[:, :, 0], joint_state[:, :, 1]
        )
 
        # ── Convert foot world positions → robot base-local frame ─────────────
        # IsaacLab convention: subtract_frame_transforms(a_pos, a_quat, b_pos)
        # gives b_pos in frame a.
        foot_local = []
        for leg_i in range(n_legs):
            # fp[:, leg_i, :]  world-frame foot target
            f_world = fp[:, leg_i, :]          # (E, 3)
            f_local, _ = subtract_frame_transforms(rp, rr, f_world)
            foot_local.append(f_local)         # (E, 3)
 
        # ── IK iteration loop ─────────────────────────────────────────────────
        for _ in range(self.ik_iters):
            # Refresh articulation data (body positions etc.)
            self.robot.update(self.physics_dt)
 
            for leg_i in range(n_legs):
                ctrl = self.ik_controllers[leg_i]
                leg_ids = self.leg_joint_ids[leg_i]
                foot_id = self.foot_body_ids[leg_i]
 
                # Current EE state in world frame
                ee_pos_w   = self.robot.data.body_pos_w[:, foot_id, :]    # (E,3)
                ee_quat_w  = self.robot.data.body_quat_w[:, foot_id, :]   # (E,4)
 
                # Convert to base-local
                ee_pos_b, ee_quat_b = subtract_frame_transforms(rp, rr, ee_pos_w, ee_quat_w)
 
                # Target is fixed in base-local frame
                target_pos_b = foot_local[leg_i]   # (E,3)
 
                # Set IK targets (position only)
                ctrl.set_command(target_pos_b)
 
                # Current leg joint positions
                leg_jpos = jpos[:, leg_ids]         # (E, 3)
 
                # Jacobian — Isaac expects (body_id, joint_ids) as integers
                jacobian_full = self.robot.root_physx_view.get_jacobians()
                # jacobian_full: (E, n_bodies-1, 6, n_dofs)
                # Slice relevant DoFs for this leg
                body_idx = foot_id - 1              # jacobian is body-1 indexed
                leg_dof_ids = torch.tensor(leg_ids, device=dev)
                J = jacobian_full[:, body_idx, :3, :][:, :, leg_dof_ids]  # (E,3,3)
 
                # Compute joint delta via controller
                delta_jpos = ctrl.compute(
                    jacobian=J,
                    current_pose=(ee_pos_b, ee_quat_b),
                    joint_pos=leg_jpos,
                )
                # Update leg joint positions
                jpos[:, leg_ids] += delta_jpos
 
                # Clamp to joint limits
                lo = self.robot.data.soft_joint_pos_limits[:, leg_ids, 0]
                hi = self.robot.data.soft_joint_pos_limits[:, leg_ids, 1]
                jpos[:, leg_ids] = torch.clamp(jpos[:, leg_ids], lo, hi)
 
            # Write updated joint positions back to sim
            self.robot.write_joint_position_to_sim(jpos)
 
        # ── Final joint velocities via articulation data ──────────────────────
        self.robot.update(self.physics_dt)
        jvel_final = self.robot.data.joint_vel   # (E, 12)
 
        return (
            jpos[0].cpu().numpy().astype(np.float32),
            jvel_final[0].cpu().numpy().astype(np.float32),
        )