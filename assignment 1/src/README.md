# An Example of Learning a Neural Network Controller
In this example, a PID controller is design for ship heading control. Then a dataset of heading control demonstration data is generated. We use these demonstration data (state-action pair) to learn a neural network controller.

## Code structure

* The code directory is as follows:
    ```
    heading_control_example/
    └───controller/
    │   │   base_controller.py
    │   │   heading_nn_controller.py
    │   │   heading_pid_controller.py
    └───sim_env/
    │   └───forces_data/
    │   │   │   rpm.txt
    │   │   │   ruffer.txt
    │   │   actuator.py
    │   │   ship_model.py
    │   │   simulator.py
    │   evaluate.py
    │   gen_data.py
    │   train.py
    │   visualize_data.py
    ```

## Usage
* Do not modify anything in `sim_env/`
* Generate data, it will generate data in `demonstration_data/`
```
python gen_data.py
```
* (Optinal) Visualize the generated trajectory data
```
python visualize_data.py
```
* Train the model
```
python train.py
```
* Evaluate the model in a close loop control
```
python evaluate.py
```