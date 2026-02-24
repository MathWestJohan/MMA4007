import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent   # folder containing visualize_data.py
path = BASE_DIR / "demonstration_data"


def plot_file(df, fig_title=None):
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    if fig_title:
        fig.suptitle(fig_title)

    # Trajectory
    ax[0, 0].plot(df['x'], df['y'])
    ax[0, 0].scatter([df['x'].iloc[0]], [df['y'].iloc[0]], label='start')
    ax[0, 0].set_title('Trajectory')
    ax[0, 0].set_xlabel('x (m)')
    ax[0, 0].set_ylabel('y (m)')
    ax[0, 0].legend()

    # Heading (convert rad -> deg)
    ax[0, 1].plot(df['t'], np.rad2deg(df['psi']))
    ax[0, 1].set_title('Heading')
    ax[0, 1].set_xlabel('time (s)')
    ax[0, 1].set_ylabel('Heading (deg)')

    # Surge speed
    ax[0, 2].plot(df['t'], df['u'])
    ax[0, 2].set_title('Surge speed')
    ax[0, 2].set_xlabel('time (s)')
    ax[0, 2].set_ylabel('u (m/s)')

    # Sway speed
    ax[1, 0].plot(df['t'], df['v'])
    ax[1, 0].set_title('Sway speed')
    ax[1, 0].set_xlabel('time (s)')
    ax[1, 0].set_ylabel('v (m/s)')

    # Use actual values if available, otherwise fall back to commanded
    rpm_col = 'act_rpm' if 'act_rpm' in df.columns else 'cmd_rpm'
    rudder_col = 'act_rudder' if 'act_rudder' in df.columns else 'cmd_rudder'

    ax[1, 1].plot(df['t'], df[rpm_col])
    ax[1, 1].set_title('Actual RPM' if rpm_col == 'act_rpm' else 'Commanded RPM')
    ax[1, 1].set_xlabel('time (s)')
    ax[1, 1].set_ylabel('RPM')

    ax[1, 2].plot(df['t'], df[rudder_col])
    ax[1, 2].set_title('Actual rudder' if rudder_col == 'act_rudder' else 'Commanded rudder')
    ax[1, 2].set_xlabel('time (s)')
    ax[1, 2].set_ylabel('Angle (deg)')

    plt.tight_layout()
    plt.show()


def plot_all(path):
    files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    for file in files:
        df = pd.read_csv(os.path.join(path, file))

        ax[0, 0].plot(df['x'], df['y'])
        ax[0, 1].plot(df['t'], np.rad2deg(df['psi']))   # fixed units
        ax[0, 2].plot(df['t'], df['u'])
        ax[1, 0].plot(df['t'], df['v'])

        rpm_col = 'act_rpm' if 'act_rpm' in df.columns else 'cmd_rpm'
        rudder_col = 'act_rudder' if 'act_rudder' in df.columns else 'cmd_rudder'
        ax[1, 1].plot(df['t'], df[rpm_col])
        ax[1, 2].plot(df['t'], df[rudder_col])

    ax[0, 0].set_title('Trajectory')
    ax[0, 0].set_xlabel('x (m)')
    ax[0, 0].set_ylabel('y (m)')

    ax[0, 1].set_title('Heading')
    ax[0, 1].set_xlabel('time (s)')
    ax[0, 1].set_ylabel('Heading (deg)')

    ax[0, 2].set_title('Surge speed')
    ax[0, 2].set_xlabel('time (s)')
    ax[0, 2].set_ylabel('u (m/s)')

    ax[1, 0].set_title('Sway speed')
    ax[1, 0].set_xlabel('time (s)')
    ax[1, 0].set_ylabel('v (m/s)')

    ax[1, 1].set_title('RPM')
    ax[1, 1].set_xlabel('time (s)')
    ax[1, 1].set_ylabel('RPM')

    ax[1, 2].set_title('Rudder angle')
    ax[1, 2].set_xlabel('time (s)')
    ax[1, 2].set_ylabel('Angle (deg)')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    path = BASE_DIR / "demonstration_data"
    files = sorted([f.name for f in path.iterdir() if f.suffix == '.csv'])

    mode = input("Choose mode ('single' or 'all') [single]: ").strip().lower()

    if mode in ("", "single", "s"):
        if not files:
            print("No CSV files found in demonstration_data.")
        else:
            print("\nAvailable files:")
            for i, f in enumerate(files):
                print(f"[{i}] {f}")

            choice = input("\nEnter file index (blank for 0): ").strip()
            idx = 0 if choice == "" else int(choice)
            idx = max(0, min(idx, len(files) - 1))

            file_path = os.path.join(path, files[idx])
            df = pd.read_csv(file_path)
            plot_file(df, fig_title=f"Dataset visualization: {files[idx]}")

    elif mode in ("all", "a"):
        plot_all(path)

    else:
        print("Unknown mode. Use 'single' or 'all'.")