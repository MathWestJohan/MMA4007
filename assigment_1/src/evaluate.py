from controller.heading_nn_controller import RudderNNController
from sim_env.simulator import Sim, ShipState
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def run_case(init_deg, tgt_deg, tf=200, dt=1.0, model_path="rudder_model.pt"):
    state0 = ShipState(500, 400, np.deg2rad(init_deg), 0, 0, 0)
    target = ShipState(0, 0, np.deg2rad(tgt_deg), 10, 0, 0)

    nn = RudderNNController(model_path=model_path, default_rpm=80.0)
    sim = Sim(state0, target, nn, dt=dt, tf=tf)
    df = pd.DataFrame(sim.run())

    # heading error metrics
    e_psi = wrap_to_pi(target.psi - df["psi"].values)
    final_err_deg = np.rad2deg(e_psi[-1])
    mae_deg = np.rad2deg(np.mean(np.abs(e_psi)))

    return df, final_err_deg, mae_deg


def plot_single_case(df, init_deg, tgt_deg):
    plt.figure()
    plt.plot(df["x"], df["y"], label=f"{init_deg}->{tgt_deg}")
    plt.scatter([df["x"].iloc[0]], [df["y"].iloc[0]], marker="o", label="start")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"NN trajectory ({init_deg}° -> {tgt_deg}°)")
    plt.axis("equal")
    plt.legend()
    plt.show()


def benchmark_mode():
    cases = [
        (180, 40),
        (0, 90),
        (270, 45),
        (30, 210),
        (350, 10),
        (120, 300),
    ]

    plt.figure()
    rows = []

    for init_deg, tgt_deg in cases:
        df, final_err_deg, mae_deg = run_case(init_deg, tgt_deg)
        plt.plot(df["x"], df["y"], label=f"{init_deg}->{tgt_deg}")
        rows.append({
            "init_psi_deg": init_deg,
            "target_psi_deg": tgt_deg,
            "final_heading_error_deg": final_err_deg,
            "mean_abs_heading_error_deg": mae_deg,
        })

    print(pd.DataFrame(rows))

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("NN trajectories across multiple heading targets")
    plt.axis("equal")
    plt.show()


def single_mode():
    try:
        init_deg_str = input("Enter initial heading in degrees (e.g. 180): ").strip()
        tgt_deg_str = input("Enter target heading in degrees (e.g. 40): ").strip()

        init_deg = float(init_deg_str)
        tgt_deg = float(tgt_deg_str)
    except ValueError:
        print("Invalid input. Please enter numeric angles (degrees).")
        return

    df, final_err_deg, mae_deg = run_case(init_deg, tgt_deg)

    print("\nSingle-case NN evaluation")
    print(f"Initial heading: {init_deg:.1f} deg")
    print(f"Target heading:  {tgt_deg:.1f} deg")
    print(f"Final heading error:      {final_err_deg:.3f} deg")
    print(f"Mean abs heading error:   {mae_deg:.3f} deg")

    plot_single_case(df, init_deg, tgt_deg)


if __name__ == "__main__":
    mode = input("Choose mode ('single' or 'benchmark') [single]: ").strip().lower()

    if mode in ("", "single", "s"):
        single_mode()
    elif mode in ("benchmark", "b"):
        benchmark_mode()
    else:
        print("Unknown mode. Use 'single' or 'benchmark'.")