import pandas as pd
import os
import matplotlib.pyplot as plt


def plot(path):
    files = os.listdir(path)

    fig, ax = plt.subplots(2, 3, figsize=(12,8))

    for file in files:
        df = pd.read_csv(os.path.join(path, file))
        ax[0, 0].plot(df['x'], df['y'])
        ax[0, 0].set_title('Trajectory')
        ax[0, 0].set_xlabel('x (m)')
        ax[0, 0].set_ylabel('y (m)')

        ax[0, 1].plot(df['t'], df['psi'])
        ax[0, 1].set_title('Heading')
        ax[0, 1].set_xlabel('time (s)')
        ax[0, 1].set_ylabel('Heading (deg)')

        ax[0, 2].plot(df['t'], df['u'])
        ax[0, 2].set_title('Surge speed')
        ax[0, 2].set_xlabel('time (s)')
        ax[0, 2].set_ylabel('u (m/s)')

        ax[1, 0].plot(df['t'], df['v'])
        ax[1, 0].set_title('Sway speed')
        ax[1, 0].set_xlabel('time (s)')
        ax[1, 0].set_ylabel('v (m/s)')

        ax[1, 1].plot(df['t'], df['cmd_rpm'])
        ax[1, 1].set_title('Actual RPM')
        ax[1, 1].set_xlabel('time (s)')
        ax[1, 1].set_ylabel('RPM')

        ax[1, 2].plot(df['t'], df['cmd_rudder'])
        ax[1, 2].set_title('Actual rudder')
        ax[1, 2].set_xlabel('time (s)')
        ax[1, 2].set_ylabel('Angle (deg)')

    plt.show()

if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), "demonstration_data")
    plot(path)