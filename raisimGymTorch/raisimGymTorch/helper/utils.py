import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
import os

def plot_masses(names: List[str],
                masses: List[float],
                title: Optional[str] = None,
                figsize: tuple = (10, 6),
                rotate: int = 45,
                annotate: bool = True,
                horizontal: bool = False,
                save_path: Optional[str] = None):
    assert len(names) == len(masses), "names and masses must have same length"
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        bars = ax.barh(x, masses, color='C0')
        ax.set_yticks(x)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel('Mass')
    else:
        bars = ax.bar(x, masses, color='C0')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=rotate, ha='right', fontsize=10)
        ax.set_ylabel('Mass')

    if title:
        ax.set_title(title)

    if annotate:
        for b in bars:
            if horizontal:
                val = b.get_width()
                ax.annotate(f'{val:.3g}', xy=(val, b.get_y() + b.get_height() / 2),
                            xytext=(3, 0), textcoords='offset points', va='center', ha='left', fontsize=9)
            else:
                val = b.get_height()
                ax.annotate(f'{val:.3g}', xy=(b.get_x() + b.get_width() / 2, val),
                            xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()
    plt.close(fig)

def plot_force_trajectories(force_data: np.ndarray,
                            title: Optional[str] = None,
                            figsize: tuple = (12, 6),
                            obj_name = None,
                            save_path: Optional[str] = None):
    """
    Plots force trajectories for multiple environments.

    Args:
        force_data (np.ndarray): 2D array where each row corresponds to an environment's force trajectory over time.
        title (Optional[str]): Title of the plot.
        figsize (tuple): Size of the figure.
        save_path (Optional[str]): If provided, saves the plot to this path.
    """
    
    num_steps, num_envs = force_data.shape
    time = np.arange(num_steps)
    print("force_data shape:", force_data.shape)

    plt.figure(figsize=figsize)
    for i in range(num_envs):
        if obj_name:
            plt.plot(time, force_data[:, i], label=f'Env {i+1} ({obj_name[i]})')
        else:
            plt.plot(time, force_data[:, i])
        print("Obj Name:", obj_name[i] if obj_name else "N/A")
        plt.show()

    plt.xlabel('Time Step')
    plt.ylabel('Force')
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
# Obj Name: cracker_box_oriented
# Obj Name: 003_cracker_box
# Obj Name: 036_wood_block
# Obj Name: 021_bleach_cleanser

    if save_path:
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()
    plt.close()

# 使用示例
if __name__ == '__main__':
    names = ['obj_a', 'obj_b', 'obj_c']
    masses = [0.12, 1.5, 0.75]
    plot_masses(names, masses, title='Object Masses', save_path='masses.png')