# Joint limits from URDF soft limits
# Order matches joint index: [Revolute1, Revolute2, Revolute3, Revolute4, Revolute5, Revolute6]
JOINT_LOWER = [-0.2618, -0.2618, -0.3490, -0.3490, -0.3490, -0.3490]
JOINT_UPPER = [ 0.0872,  0.0872,  0.1745,  0.1745,  0.2618,  0.2618]

# Derived — do not edit directly
JOINT_MID        = [(l + u) / 2 for l, u in zip(JOINT_LOWER, JOINT_UPPER)]
JOINT_HALF_RANGE = [(u - l) / 2 for l, u in zip(JOINT_LOWER, JOINT_UPPER)]
JOINT_FULL_RANGE = [(u - l)     for l, u in zip(JOINT_LOWER, JOINT_UPPER)]

# Per-stepper addr (1-4 → Revolute3-6) joint limits for rs485 clamping
STEPPER_LIMITS = {
    1: (JOINT_LOWER[2], JOINT_UPPER[2]),  # Revolute3 (hip)
    2: (JOINT_LOWER[3], JOINT_UPPER[3]),  # Revolute4 (hip)
    3: (JOINT_LOWER[4], JOINT_UPPER[4]),  # Revolute5 (knee)
    4: (JOINT_LOWER[5], JOINT_UPPER[5]),  # Revolute6 (knee)
}
