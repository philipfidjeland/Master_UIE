# Run Customer Satisfaction Network

1. **This application should be run inside Vitis-AI-RNN docker container.**
    ```sh
    cd ${VAI_HOME}/demo/DPU-for-RNN/rnn_u25_u50lv/apps/customer_satisfaction/
    ```

1. Setup the environment
    ```sh
    # For U25
    TARGET_DEVICE=U25 source ./setup.sh

    # For U50LV
    TARGET_DEVICE=U50LV source ./setup.sh
    ```

1. Run the cpu mode with all trasactions
    ```sh
    python run_cpu_e2e.py
    ```

1. Run the dpu mode with all transactions
    ```sh
    # For U25
    TARGET_DEVICE=U25 python run_dpu_e2e.py

    # For U50LV
    TARGET_DEVICE=U50LV python run_dpu_e2e.py
    ```

1. Run the dpu mode with multithread support
    ```sh
    # For U25
    TARGET_DEVICE=U25 python run_dpu_e2e_mt.py -n 4

    # For U50LV
    TARGET_DEVICE=U50LV python run_dpu_e2e_mt.py -n 4
    ```

1. The accuracy for the end-to-end model test should be around `0.9565`

#### License
Copyright 2021 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
