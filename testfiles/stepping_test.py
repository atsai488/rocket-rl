import argparse
import time
from drivers.rs485 import Rs485Driver
from rocket.rocket import JOINT_POS_MID, JOINT_SCALE

# Stepper joints only (indices 2-5): [HipL, HipR, KneeL, KneeR]
STEPPER_MIDS   = JOINT_POS_MID[2:]
STEPPER_SCALES = JOINT_SCALE[2:]

# Policy outputs in [-1, 1]: [hip_L, hip_R, knee_L, knee_R]
PHASES = [
    {"name": "1_R_LIFT",       "policy": [ 0.55,  0.81,  0.95, -0.95]},  # L planted, R lifts
    {"name": "2_R_TRANSFER",   "policy": [ 0.56,  0.19,  0.95,  0.95]},  # R lands
    {"name": "3_L_LIFT",       "policy": [ 0.96, -0.27, -0.95,  0.95]},  # R planted, L lifts
    {"name": "4_L_TRANSFER",   "policy": [ 0.56,  0.19,  0.95,  0.95]},  # L lands → back to phase 1
]


def scale_policy(policy_val: float, mid: float, scale_: float) -> float:
    return max(mid - scale_, min(mid + policy_val * scale_, mid + scale_))


def policy_to_joints(policy: list) -> dict:
    targets = [scale_policy(p, m, s) for p, m, s in zip(policy, STEPPER_MIDS, STEPPER_SCALES)]
    return {addr: targets[addr - 1] for addr in range(1, 5)}


def main():
    parser = argparse.ArgumentParser(description="Open-loop stepping gait test")
    parser.add_argument("--hz",          type=float, default=10.0, help="Send rate in Hz (default 10)")
    parser.add_argument("--cycles",      type=int,   default=5,    help="Number of full gait cycles (default 5)")
    parser.add_argument("--hold-cycles", type=int,   default=3,    help="Sends per phase to let motors converge (default 3)")
    parser.add_argument("--max-pulses",  type=int,   default=5,   help="Max pulses per send (default 20, keep low for safety)")
    parser.add_argument("--run",         action="store_true",       help="Actually send commands to motors (dry-run by default)")
    args = parser.parse_args()

    period_s = 1.0 / args.hz
    driver = Rs485Driver() if args.run else None

    print(f"Stepping test — {args.hz}Hz  gait_cycles={args.cycles}  hold_cycles={args.hold_cycles}  max_pulses={args.max_pulses}")
    print(f"Mids:   {[f'{m:.4f}' for m in STEPPER_MIDS]}")
    print(f"Scales: {[f'{s:.4f}' for s in STEPPER_SCALES]}")
    print(f"Max reach per send: {args.max_pulses / (200 * 5) * 6.2832:.4f} rad")
    print(f"Mode: {'LIVE — sending to motors' if args.run else 'DRY RUN — pass --run to send'}\n")
    input("Press ENTER to start...")

    try:
        for cycle in range(args.cycles):
            for phase in PHASES:
                joints = policy_to_joints(phase["policy"])
                _NAMES = {1: "Hip L", 2: "Hip R", 3: "Knee L", 4: "Knee R"}
                for hold in range(args.hold_cycles):
                    print(f"\n[Cycle {cycle+1}/{args.cycles}] Phase={phase['name']} hold={hold+1}/{args.hold_cycles}")
                    for addr, rad in joints.items():
                        print(f"  {_NAMES[addr]}: {rad:.4f} rad")
                    t0 = time.time()
                    if args.run:
                        driver.send_joint_position(joints, max_pulses=args.max_pulses)
                    elapsed = time.time() - t0
                    leftover = period_s - elapsed
                    if leftover > 0:
                        time.sleep(leftover)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if driver:
            driver.close()
        print("Done.")


if __name__ == "__main__":
    main()
