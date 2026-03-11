from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.imitation_env_cfg import ImitationRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip

import copy

# Import sim utils to disable the physics for Ghost Robot
import isaaclab.sim as sim_utils


@configclass
class AnymalDImitationRoughEnvCfg(ImitationRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-d
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.ghost_robot = copy.deepcopy(ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/GhostRobot"))


        # 1. Disable gravity and collisions (Do NOT make it kinematic)
        self.scene.ghost_robot.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True
        )
        self.scene.ghost_robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
            collision_enabled=False
        )
        
        # 2. Apply a transparent material to the entire USD
        # self.scene.ghost_robot.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
        #     diffuse_color=(0.2, 0.5, 0.8), # Holographic blue tint
        #     opacity=0.4 
        # )

        if hasattr(self.scene.ghost_robot, "actuators"):
            for actuator_name in self.scene.ghost_robot.actuators.keys():
                self.scene.ghost_robot.actuators[actuator_name].stiffness = 0.0
                self.scene.ghost_robot.actuators[actuator_name].damping = 0.0


@configclass
class AnymalDImitationRoughEnvCfg_PLAY(AnymalDImitationRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.5    # original: 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class AnymalDImitationFlatEnvCfg(AnymalDImitationRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.feet_air_time.weight = 0.5
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class AnymalDImitationFlatEnvCfg_PLAY(AnymalDImitationFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
