"""GIF Creator - Handles animated GIF creation for diffusion model trajectories."""

from pathlib import Path
from typing import List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch


class GifCreator:
    """Creates animated GIFs for diffusion model denoising trajectories."""

    def __init__(self, figsize: Tuple[int, int] = (10, 8), fps: int = 4, dpi: int = 100) -> None:
        """Initialize GIF creator.

        Args:
            figsize: Figure size for animation frames
            fps: Frames per second for the GIF
            dpi: DPI for the animation
        """
        self.figsize = figsize
        self.fps = fps
        self.dpi = dpi
        self.default_style = {
            "xlim": (-2.5, 2.5),
            "ylim": (-2.5, 2.5),
            "grid_alpha": 0.3,
            "original_alpha": 0.3,
            "original_size": 10,
            "sample_alpha": 0.3,
            "sample_size": 20,
        }

    def create_trajectory_gif(
        self,
        trajectory: List[torch.Tensor],
        original_data: torch.Tensor,
        output_path: Path,
        n_frames: int = 20,
        title_template: str = "Denoising Diffusion Process - Step {step}/{total}",
    ) -> Path:
        """Create a GIF animation of the denoising trajectory.

        Args:
            trajectory: List of tensors representing denoising steps
            original_data: Original training data for background
            output_path: Path to save the GIF
            n_frames: Number of frames to sample from trajectory
            title_template: Template for frame titles

        Returns:
            Path: Path to the created GIF file
        """
        print(f"Creating trajectory GIF with {n_frames} frames...")

        # Sample frames from trajectory
        total_steps = len(trajectory)
        if n_frames > total_steps:
            n_frames = total_steps

        frame_indices = np.linspace(0, total_steps - 1, n_frames, dtype=int)
        sampled_trajectory = [trajectory[i] for i in frame_indices]

        print(f"Sampled {n_frames} frames from {total_steps} total timesteps")
        print(f"Frame indices: {frame_indices}")

        # Convert data to numpy
        data_np = original_data.numpy() if torch.is_tensor(original_data) else original_data

        # Set up figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(self.default_style["xlim"])
        ax.set_ylim(self.default_style["ylim"])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=self.default_style["grid_alpha"])

        # Plot original data as background
        ax.scatter(
            data_np[:, 0],
            data_np[:, 1],
            alpha=self.default_style["original_alpha"],
            s=self.default_style["original_size"],
            c="blue",
            label="Original Data",
        )

        # Initialize scatter plot for animated points
        scat = ax.scatter(
            [],
            [],
            s=self.default_style["sample_size"],
            c="red",
            alpha=self.default_style["sample_alpha"],
            label="Generated Samples",
        )

        # Add title and legend
        title = ax.set_title(title_template.format(step=0, total=total_steps - 1))
        ax.legend(loc="upper right")

        def animate(frame):
            """Animation function for each frame."""
            current_traj = sampled_trajectory[frame]
            current_traj_np = (
                current_traj.numpy() if torch.is_tensor(current_traj) else current_traj
            )
            actual_timestep = frame_indices[frame]

            # Update scatter plot
            scat.set_offsets(current_traj_np)

            # Update title
            title.set_text(title_template.format(step=actual_timestep, total=total_steps - 1))

            return scat, title

        # Create animation
        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=len(sampled_trajectory),
            interval=1000 // self.fps,
            blit=False,
            repeat=True,
        )

        # Save as GIF
        print(f"Saving GIF to {output_path}...")
        anim.save(str(output_path), writer="pillow", fps=self.fps, dpi=self.dpi)
        plt.close(fig)

        print(f"GIF saved successfully! {len(sampled_trajectory)} frames at {self.fps} FPS")
        return output_path

    def create_side_by_side_gif(
        self,
        trajectory: List[torch.Tensor],
        original_data: torch.Tensor,
        output_path: Path,
        n_frames: int = 20,
        title_template: str = "Denoising Process - Step {step}",
    ) -> Path:
        """Create a side-by-side GIF showing original data and current denoising step.

        Args:
            trajectory: List of tensors representing denoising steps
            original_data: Original training data
            output_path: Path to save the GIF
            n_frames: Number of frames to sample from trajectory
            title_template: Template for frame titles

        Returns:
            Path: Path to the created GIF file
        """
        print(f"Creating side-by-side trajectory GIF with {n_frames} frames...")

        # Sample frames from trajectory
        total_steps = len(trajectory)
        if n_frames > total_steps:
            n_frames = total_steps

        frame_indices = np.linspace(0, total_steps - 1, n_frames, dtype=int)
        sampled_trajectory = [trajectory[i] for i in frame_indices]

        # Convert data to numpy
        data_np = original_data.numpy() if torch.is_tensor(original_data) else original_data

        # Set up figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Original data subplot (static)
        ax1.scatter(data_np[:, 0], data_np[:, 1], alpha=0.6, s=20, c="blue")
        ax1.set_title("Original Data")
        ax1.set_xlim(self.default_style["xlim"])
        ax1.set_ylim(self.default_style["ylim"])
        ax1.grid(True, alpha=self.default_style["grid_alpha"])

        # Generated samples subplot (animated)
        ax2.set_xlim(self.default_style["xlim"])
        ax2.set_ylim(self.default_style["ylim"])
        ax2.grid(True, alpha=self.default_style["grid_alpha"])

        # Initialize scatter plot for animated points
        scat = ax2.scatter([], [], s=20, c="red", alpha=0.8)
        title2 = ax2.set_title(title_template.format(step=0))

        def animate(frame):
            """Animation function for each frame."""
            current_traj = sampled_trajectory[frame]
            current_traj_np = (
                current_traj.numpy() if torch.is_tensor(current_traj) else current_traj
            )
            actual_timestep = frame_indices[frame]

            # Update scatter plot
            scat.set_offsets(current_traj_np)

            # Update title
            title2.set_text(title_template.format(step=actual_timestep))

            return scat, title2

        # Create animation
        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=len(sampled_trajectory),
            interval=1000 // self.fps,
            blit=False,
            repeat=True,
        )

        # Save as GIF
        print(f"Saving side-by-side GIF to {output_path}...")
        anim.save(str(output_path), writer="pillow", fps=self.fps, dpi=self.dpi)
        plt.close(fig)

        print(f"Side-by-side GIF saved successfully!")
        return output_path

    def create_progression_gif(
        self,
        trajectory: List[torch.Tensor],
        original_data: torch.Tensor,
        output_path: Path,
        n_frames: int = 20,
        fade_previous: bool = True,
    ) -> Path:
        """Create a GIF showing progression with fading previous steps.

        Args:
            trajectory: List of tensors representing denoising steps
            original_data: Original training data for background
            output_path: Path to save the GIF
            n_frames: Number of frames to sample from trajectory
            fade_previous: Whether to show fading previous steps

        Returns:
            Path: Path to the created GIF file
        """
        print(f"Creating progression GIF with {n_frames} frames...")

        # Sample frames from trajectory
        total_steps = len(trajectory)
        if n_frames > total_steps:
            n_frames = total_steps

        frame_indices = np.linspace(0, total_steps - 1, n_frames, dtype=int)
        sampled_trajectory = [trajectory[i] for i in frame_indices]

        # Convert data to numpy
        data_np = original_data.numpy() if torch.is_tensor(original_data) else original_data

        # Set up figure
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(self.default_style["xlim"])
        ax.set_ylim(self.default_style["ylim"])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=self.default_style["grid_alpha"])

        def animate(frame):
            """Animation function for each frame."""
            ax.clear()
            ax.set_xlim(self.default_style["xlim"])
            ax.set_ylim(self.default_style["ylim"])
            ax.grid(True, alpha=self.default_style["grid_alpha"])

            # Plot original data as background
            ax.scatter(
                data_np[:, 0],
                data_np[:, 1],
                alpha=self.default_style["original_alpha"],
                s=8,
                c="blue",
                label="Original",
            )

            # Plot previous steps with fading if enabled
            if fade_previous and frame > 0:
                for i in range(max(0, frame - 3), frame):
                    prev_traj = sampled_trajectory[i]
                    prev_traj_np = prev_traj.numpy() if torch.is_tensor(prev_traj) else prev_traj
                    alpha_val = 0.2 + 0.2 * (i - max(0, frame - 3)) / max(1, frame - 1)
                    ax.scatter(
                        prev_traj_np[:, 0],
                        prev_traj_np[:, 1],
                        alpha=alpha_val,
                        s=10,
                        c="orange",
                    )

            # Plot current step
            current_traj = sampled_trajectory[frame]
            current_traj_np = (
                current_traj.numpy() if torch.is_tensor(current_traj) else current_traj
            )
            actual_timestep = frame_indices[frame]

            ax.scatter(
                current_traj_np[:, 0],
                current_traj_np[:, 1],
                s=20,
                c="red",
                alpha=0.5,
                label="Current",
            )

            ax.set_title(f"Denoising Progression - Step {actual_timestep}/{total_steps-1}")
            ax.legend()

        # Create animation
        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=len(sampled_trajectory),
            interval=1000 // self.fps,
            repeat=True,
        )

        # Save as GIF
        print(f"Saving progression GIF to {output_path}...")
        anim.save(str(output_path), writer="pillow", fps=self.fps, dpi=self.dpi)
        plt.close(fig)

        print(f"Progression GIF saved successfully!")
        return output_path

    def set_style(self, style_dict: dict) -> None:
        """Update default style settings.

        Args:
            style_dict: Dictionary with style parameters to update
        """
        self.default_style.update(style_dict)
