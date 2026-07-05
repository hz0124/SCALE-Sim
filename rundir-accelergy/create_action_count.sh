#!/bin/bash

python3 create_action_count.py --saved_folder /tmp/argus_f7_lmrxj886/scsim_log --run_name scale_example_run_32x32_os --arch_name systolic_array --SRAM_row_size 5 --DRAM_row_size 5 --config /tmp/argus_f7_lmrxj886/hw_cfg_with_accelergy.cfg

cp /tmp/argus_f7_lmrxj886/scsim_log/scale_example_run_32x32_os/action_count.yaml ./accelergy_input/action_count.yaml

mv /tmp/argus_f7_lmrxj886/scsim_log/scale_example_run_32x32_os  /tmp/argus_f7_lmrxj886/all_output/scale_sim_output_scale_example_run_32x32_os

