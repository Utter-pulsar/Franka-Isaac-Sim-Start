import torch




class To_Position():
    def __init__(self, ik_commands, robot_actions, device, controller, envs_positions, robot) -> None:
        self.ik_commands = ik_commands
        self.robot_actions = robot_actions
        self.device = device
        self.controller = controller
        self.envs_positions = envs_positions
        self.robot = robot



    def execute(self, position = [0.5, 0, 0.5, 0, 1, 0, 0]):
        ee_goals = torch.tensor(position, device=self.device)
        # Track the given command
        self.ik_commands[:] = ee_goals
        # set the controller commands
        self.controller.set_command(self.ik_commands)
        # compute the joint commands
        self.robot_actions[:, : self.robot.arm_num_dof] = self.controller.compute(
            self.robot.data.ee_state_w[:, 0:3] - self.envs_positions,
            self.robot.data.ee_state_w[:, 3:7],
            self.robot.data.ee_jacobian,
            self.robot.data.arm_dof_pos,
        )
        # in some cases the zero action correspond to offset in actuators
        # so we need to subtract these over here so that they can be added later on
        arm_command_offset = self.robot.data.actuator_pos_offset[:, : self.robot.arm_num_dof]
        # offset actuator command with position offsets
        # note: valid only when doing position control of the robot
        self.robot_actions[:, : self.robot.arm_num_dof] -= arm_command_offset
        # apply actions
        self.robot.apply_action(self.robot_actions)

    def gripper(self, is_Open = True):
        if is_Open == True:
            self.robot_actions[:,-1] = 1
        else:
            self.robot_actions[:,-1] = -1