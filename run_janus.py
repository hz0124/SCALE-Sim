import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulation_core.janus_sim import Janus_sim
def main():
    """主函数"""

    parser = argparse.ArgumentParser(description="Run Bagel Simulation")

    parser.add_argument('--hw', type=str, default='ours', 
                        choices=['ours', 'ours_balence', 'sdma', 'flightvgm', 'figna'],
                        help='Hardware architecture type')
    
    parser.add_argument('--task', type=str, default='GenEval',
                        choices=['GenEval', 'MM'],
                        help='Task/Dataset type')
    
    parser.add_argument('--config', type=str, default=None,
                        help='Optional: Path to a specific .json file, to config the hole simulation run')

    parser.add_argument('--energy', action='store_true',
                        help='Enable energy accounting in results.log (~1.5x runtime). '
                             'Coefficients come from simulation_core/energy_accounting/coefficients.py '
                             'and can be overridden in the workload JSON via "energy_coefficients" '
                             'and "dram_type" fields.')

    parser.add_argument('--validate', action='store_true',
                        help='With --energy, also call accelergy CLI per sub-run to '
                             'cross-validate the EnergyAccountant numbers. Slow.')

    args = parser.parse_args()
    print(f"Starting Bagel Simulation...")
    print(f"Hardware: {args.hw}")
    print(f"Task:     {args.task}")
    print(f"Config:    {args.config}")
    if args.energy:
        print(f"Energy:   enabled" + ("  (+ accelergy validation)" if args.validate else ""))

    # 实例化Bagel仿真类
    janus = Janus_sim(hardware_type=args.hw, task=args.task)

    # 读取JSON配置
    janus.read_from_json(cfg_path=args.config)

    if args.energy:
        janus.energy_enabled = True
    if janus.energy_enabled:
        janus.setup_energy()
        if args.validate:
            janus.energy_validate = True

    # 运行完整模型仿真
    total_cycles = janus.run_model()
    
    print(f"\nSimulation completed successfully!")
    print(f"Total cycles: {total_cycles}")
    print(f"Results saved to: {janus.result_path}/results.log")


if __name__ == "__main__":
    main()