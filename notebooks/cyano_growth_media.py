import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    from typing import List, Tuple, Optional
    from typing import List, Tuple, Optional, Dict, Any
    from dataclasses import dataclass
    from pathlib import Path

    import numpy as np
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    import cobra
    from cobra.io import read_sbml_model, save_json_model
    from cobra import Model, Reaction, Metabolite

    import torch
    from ax import Client, RangeParameterConfig


@app.cell
def _():
    mo.md(
        r"""
    # Optimizing Media of _Synechococcus elongatus_ UTEX 2973 for Max Growth
    This notebook aims to computationally predict the best variations of the media we are using for our cyanobacterium strain (_S. elongatus_ UTEX 2973), BG-11 media, for fastest growth.

    Following standard flux balance analysis (FBA) theory, the **objective** here is formulated as **maximizing the *flux* of** a "fake" reaction formulated purely to track **biomass accumulation**, therein serving as a proxy for growth rate of the cells.

    The parameters that we will vary and **search over** are the **maximum allowed uptake rates of each of the BG-11 media components**.
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    ## Load model

    We are using the
    >composite GSM model for both Synechococcus 7942 and Synechococcus 2973

    developed in (Mueller et. al., 2017).

    The two models are in `SBMLmodel_UTEX2973.xml` and `SBMLmodel_PCC7942.xml` corresponding to supplementary files 2 and 3 of (Mueller et. al., 2017) respectively. The latter should be the one for UTEX 2973, as opposed to PCC 7942.

    ---

    Mueller, T., Ungerer, J., Pakrasi, H. et al. Identifying the Metabolic Differences of a Fast-Growth Phenotype in Synechococcus UTEX 2973. Sci Rep 7, 41569 (2017). https://doi.org/10.1038/srep41569
    """
    )
    return


@app.cell
def _():
    mo.md(r"""_Note_: The below cell might take a bit long to run if the SBML model is big.""")
    return


@app.cell
def _():
    RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
    PROCD_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
    # RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
    # PROCD_DATA_DIR = ROOT_DIR / "data" / "processed"

    model = read_sbml_model(RAW_DATA_DIR / "SBMLmodel_UTEX2973.xml")

    # model.solver = "gurobi"
    return PROCD_DATA_DIR, model


@app.cell
def _():
    mo.md(
        r"""
    ## Finding the Media Components in the Model
    Well actually, finding the ID's of the exchange reactions in the model *corresponding to* the media components.
    """
    )
    return


@app.cell
def _():
    BG_11_MEDIA_EXRXNS_IDS = {
        "EX_CO2",
        "EX_PHO1",
        "EX_PHO2",
        "EX_Citrate",
        "EX_NH3",
        "EX_Mn2_",
        "EX_Zinc",
        "EX_Cu2_",
        "EX_Molybdate",
        "EX_Co2_",
        "EX_Nitrate",
        "EX_Phosphate",
        "EX_Sulfate",
        "EX_Fe3_",
        "EX_H2CO3",
        "EX_Calcium"
    }
    return (BG_11_MEDIA_EXRXNS_IDS,)


@app.cell
def _(BG_11_MEDIA_EXRXNS_IDS, model):
    for _rxn_id in BG_11_MEDIA_EXRXNS_IDS:
        if _rxn_id not in model.reactions:
            raise ValueError(f'Reaction {_rxn_id} not found in the model.')
        else:
            print(model.reactions.get_by_id(_rxn_id))
    return


@app.cell
def _():
    mo.md(
        r"""
    ### Oxygen
    On Mars, no oxygen $\implies v_{\mathtt{EX\_O2}} \geq 0$.
    """
    )
    return


@app.cell
def _(model):
    print(f"Default bounds on oxygen exchange reaction: {model.reactions.EX_O2.lower_bound} < v_{{EX_O2}} < {model.reactions.EX_O2.upper_bound}")
    return


@app.cell
def _():
    mo.md(
        r"""
    ## Phenotypic Phase Planes
    First explore some potentially important components, varying pairs simultaneously at a time.
    """
    )
    return


@app.cell
def _():
    @dataclass
    class PhasePlaneResult:
        """Container for phase plane analysis results."""
        X: np.ndarray
        Y: np.ndarray
        objective_matrix: np.ndarray
        best_coordinates: Tuple[float, float]
        best_objective_value: float
        x_range: Tuple[float, float]
        y_range: Tuple[float, float]
        rxnsIDs_x: List[str]
        rxnsIDs_y: List[str]
        total_evaluations: int
        search_history: List[Dict[str, Any]] = None

    def compute_phase_plane_grid(model: cobra.Model,
                                rxnsIDs_x: List[str],
                                rxnsIDs_y: List[str],
                                x_range: Tuple[float, float],
                                y_range: Tuple[float, float],
                                x_steps: int,
                                y_steps: int,
                                obj_rxn_id: Optional[str] = None) -> PhasePlaneResult:
        """
        Compute phase plane analysis for a fixed grid without plotting.

        Args:
            model: COBRApy model
            rxnsIDs_x: List of reaction IDs for x-axis
            rxnsIDs_y: List of reaction IDs for y-axis  
            x_range: Tuple (min, max) for x-axis range
            y_range: Tuple (min, max) for y-axis range
            x_steps: Number of steps for x-axis
            y_steps: Number of steps for y-axis
            obj_rxn_id: Optional objective reaction ID

        Returns:
            PhasePlaneResult object containing all results
        """

        # Input validation
        for rxn_id in rxnsIDs_x + rxnsIDs_y:
            if rxn_id not in model.reactions:
                raise ValueError(f'Reaction {rxn_id} not found in the model.')

        with model:
            # Set objective if specified
            if obj_rxn_id is not None:
                model.objective = obj_rxn_id

            # Create coordinate arrays
            x_values = np.linspace(x_range[0], x_range[1], x_steps)
            y_values = np.linspace(y_range[0], y_range[1], y_steps)
            X, Y = np.meshgrid(x_values, y_values, indexing='ij')

            # Initialize results matrix
            objective_matrix = np.zeros(X.shape)
            evaluation_count = 0

            # Iterate through all combinations
            for i, (x_val, y_val) in enumerate(zip(X.flat, Y.flat)):
                idx_2d = np.unravel_index(i, X.shape)

                # Set bounds for reactions
                for rxn_id in rxnsIDs_x:
                    model.reactions.get_by_id(rxn_id).lower_bound = x_val
                for rxn_id in rxnsIDs_y:
                    model.reactions.get_by_id(rxn_id).lower_bound = y_val

                # Optimize
                solution = model.slim_optimize(
                    error_value=np.nan, 
                    message=f"Optimization failed at x={x_val:.3f}, y={y_val:.3f}")
                # print(f"Evaluated point {i+1}/{X.size}: x={x_val:.3f}, y={y_val:.3f}, objective={solution}")
                objective_matrix[idx_2d] = solution
                evaluation_count += 1

        # Find best solution
        valid_mask = ~np.isnan(objective_matrix)
        if not np.any(valid_mask):
            best_coordinates = (None, None)
            best_objective_value = np.nan
            print("No feasible solutions found in the grid.")
        else:
            best_idx = np.nanargmax(objective_matrix)
            best_idx_2d = np.unravel_index(best_idx, objective_matrix.shape)
            best_objective_value = objective_matrix[best_idx_2d]
            best_coordinates = (X[best_idx_2d], Y[best_idx_2d])

            print(f"Best objective value: {best_objective_value:.6f} at coordinates x={best_coordinates[0]:.6f}, y={best_coordinates[1]:.6f}")

        return PhasePlaneResult(
            X=X,
            Y=Y,
            objective_matrix=objective_matrix,
            best_coordinates=best_coordinates,
            best_objective_value=best_objective_value,
            x_range=x_range,
            y_range=y_range,
            rxnsIDs_x=rxnsIDs_x,
            rxnsIDs_y=rxnsIDs_y,
            total_evaluations=evaluation_count
        )

    def hierarchical_phase_plane_search(model: cobra.Model,
                                      rxnsIDs_x: List[str],
                                      rxnsIDs_y: List[str],
                                      initial_x_range: Tuple[float, float],
                                      initial_y_range: Tuple[float, float],
                                      levels: int = 4,
                                      initial_steps: Tuple[int, int] = (100, 100),
                                      refinement_factor: float = 0.3,
                                      steps_multiplier: float = 2.2,
                                      obj_rxn_id: Optional[str] = None,
                                      min_improvement: float = 1e-6,
                                      verbose: bool = True) -> PhasePlaneResult:
        """
        Hierarchical grid search that progressively refines the search space.

        This function starts with a coarse grid over the entire search space, identifies
        the most promising region, then zooms into that region with a finer grid. This
        process continues for the specified number of levels.

        Args:
            model: COBRApy model
            rxnsIDs_x, rxnsIDs_y: Lists of reaction IDs for x and y axes
            initial_x_range, initial_y_range: Initial search ranges
            levels: Number of hierarchical refinement levels
            initial_steps: Grid resolution for first level (x_steps, y_steps)
            refinement_factor: Fraction of current range to use for next level (0.0-1.0)
            steps_multiplier: Factor to increase grid resolution at each level
            obj_rxn_id: Optional objective reaction ID
            min_improvement: Minimum improvement required to continue refinement
            verbose: Whether to print progress

        Returns:
            PhasePlaneResult with search history
        """

        current_x_range = initial_x_range
        current_y_range = initial_y_range
        current_steps = initial_steps
        search_history = []

        best_overall_objective = -np.inf
        best_overall_coords = (0, 0)
        total_evaluations = 0

        for level in range(levels):
            if verbose:
                print(f"Level {level + 1}/{levels}:")
                print(f"  Search range: x={current_x_range}, y={current_y_range}")
                print(f"  Grid resolution: {current_steps[0]}x{current_steps[1]}")

            # Compute phase plane for current level
            result = compute_phase_plane_grid(
                model, rxnsIDs_x, rxnsIDs_y,
                current_x_range, current_y_range,
                int(current_steps[0]), int(current_steps[1]),
                obj_rxn_id
            )

            total_evaluations += result.total_evaluations

            # Store search step
            search_history.append({
                'level': level + 1,
                'x_range': current_x_range,
                'y_range': current_y_range,
                'grid_size': current_steps,
                'best_coordinates': result.best_coordinates,
                'best_objective': result.best_objective_value,
                'evaluations': result.total_evaluations
            })

            # Check if we found a feasible solution
            if np.isnan(result.best_objective_value):
                if verbose:
                    print(f"  No feasible solutions found at this level")
                break

            # Check for improvement
            improvement = result.best_objective_value - best_overall_objective
            if improvement < min_improvement and level > 0:
                if verbose:
                    print(f"  Insufficient improvement ({improvement:.2e}), stopping refinement")
                break

            # Update best overall solution
            if result.best_objective_value > best_overall_objective:
                best_overall_objective = result.best_objective_value
                best_overall_coords = result.best_coordinates

            if verbose:
                print(f"  Best objective: {result.best_objective_value:.6f} at {result.best_coordinates}")
                print(f"  Improvement: {improvement:.6f}")

            # Prepare for next level (if not the last level)
            if level < levels - 1:
                x_opt, y_opt = result.best_coordinates

                # Calculate current range sizes
                x_range_size = current_x_range[1] - current_x_range[0]
                y_range_size = current_y_range[1] - current_y_range[0]

                # Calculate new range sizes (refined)
                new_x_range_size = x_range_size * refinement_factor
                new_y_range_size = y_range_size * refinement_factor

                # Center new ranges around the optimum
                new_x_min = x_opt - new_x_range_size / 2
                new_x_max = x_opt + new_x_range_size / 2
                new_y_min = y_opt - new_y_range_size / 2
                new_y_max = y_opt + new_y_range_size / 2

                # Ensure we don't go outside the original bounds
                new_x_min = max(new_x_min, initial_x_range[0])
                new_x_max = min(new_x_max, initial_x_range[1])
                new_y_min = max(new_y_min, initial_y_range[0])
                new_y_max = min(new_y_max, initial_y_range[1])

                # Update ranges and steps for next level
                current_x_range = (new_x_min, new_x_max)
                current_y_range = (new_y_min, new_y_max)
                current_steps = (int(current_steps[0] * steps_multiplier),
                               int(current_steps[1] * steps_multiplier))

                if verbose:
                    print(f"  Refining to {refinement_factor:.1%} of current range")
                    print()

        # Create final result using the last computed grid
        final_result = result
        final_result.search_history = search_history
        final_result.total_evaluations = total_evaluations
        final_result.best_coordinates = best_overall_coords
        final_result.best_objective_value = best_overall_objective

        if verbose:
            print(f"Hierarchical search completed:")
            print(f"  Total levels: {len(search_history)}")
            print(f"  Total evaluations: {total_evaluations}")
            print(f"  Final best objective: {best_overall_objective:.6f}")
            print(f"  Final best coordinates: x={best_overall_coords[0]:.6f}, y={best_overall_coords[1]:.6f}")

        return final_result

    def adaptive_phase_plane_search(model: cobra.Model,
                                   rxnsIDs_x: List[str],
                                   rxnsIDs_y: List[str],
                                   initial_x_range: Tuple[float, float],
                                   initial_y_range: Tuple[float, float],
                                   max_x_range: Tuple[float, float],
                                   max_y_range: Tuple[float, float],
                                   initial_steps: Tuple[int, int] = (100, 100),
                                   expansion_factor: float = 1.5,
                                   max_expansions: int = 5,
                                   boundary_tolerance: float = 0.05,
                                   obj_rxn_id: Optional[str] = None,
                                   verbose: bool = True) -> PhasePlaneResult:
        """
        Adaptive expanding grid search to find optimal conditions.

        Args:
            model: COBRApy model
            rxnsIDs_x, rxnsIDs_y: Reaction IDs for x and y axes
            initial_x_range, initial_y_range: Starting search ranges
            max_x_range, max_y_range: Maximum allowed search ranges
            initial_steps: Grid resolution (x_steps, y_steps)
            expansion_factor: Factor by which to expand ranges
            max_expansions: Maximum number of expansion iterations
            boundary_tolerance: Fraction of range from boundary to consider "at boundary"
            obj_rxn_id: Optional objective reaction ID
            verbose: Whether to print progress

        Returns:
            PhasePlaneResult with search history
        """

        current_x_range = initial_x_range
        current_y_range = initial_y_range
        search_history = []

        for expansion in range(max_expansions + 1):
            if verbose:
                print(f"Expansion {expansion}: x_range={current_x_range}, y_range={current_y_range}")

            # Compute phase plane for current ranges
            result = compute_phase_plane_grid(
                model, rxnsIDs_x, rxnsIDs_y,
                current_x_range, current_y_range,
                initial_steps[0], initial_steps[1],
                obj_rxn_id
            )

            # Store search step
            search_history.append({
                'expansion': expansion,
                'x_range': current_x_range,
                'y_range': current_y_range,
                'best_coordinates': result.best_coordinates,
                'best_objective': result.best_objective_value,
                'evaluations': result.total_evaluations
            })

            # Check if we found a feasible solution
            if np.isnan(result.best_objective_value):
                if verbose:
                    print(f"  No feasible solutions found in current range")
                break

            if verbose:
                print(f"  Best objective: {result.best_objective_value:.6f} at {result.best_coordinates}")

            # Check if optimum is at boundaries
            x_opt, y_opt = result.best_coordinates
            x_min, x_max = current_x_range
            y_min, y_max = current_y_range

            # Calculate boundary thresholds
            x_tol = boundary_tolerance * (x_max - x_min)
            y_tol = boundary_tolerance * (y_max - y_min)

            # Check which boundaries the optimum is near
            at_x_min = (x_opt - x_min) <= x_tol
            at_x_max = (x_max - x_opt) <= x_tol
            at_y_min = (y_opt - y_min) <= y_tol
            at_y_max = (y_max - y_opt) <= y_tol

            # If not at any boundary, we've found a local optimum
            if not (at_x_min or at_x_max or at_y_min or at_y_max):
                if verbose:
                    print(f"  Optimum well within boundaries - search converged!")
                break

            # Prepare for next expansion
            if expansion < max_expansions:
                new_x_range = list(current_x_range)
                new_y_range = list(current_y_range)

                # Expand ranges where optimum is at boundary
                if at_x_min:
                    expansion_size = (current_x_range[1] - current_x_range[0]) * (expansion_factor - 1) / 2
                    new_x_range[0] = max(max_x_range[0], current_x_range[0] - expansion_size)
                    if verbose:
                        print(f"  Expanding x-axis lower bound")

                if at_x_max:
                    expansion_size = (current_x_range[1] - current_x_range[0]) * (expansion_factor - 1) / 2
                    new_x_range[1] = min(max_x_range[1], current_x_range[1] + expansion_size)
                    if verbose:
                        print(f"  Expanding x-axis upper bound")

                if at_y_min:
                    expansion_size = (current_y_range[1] - current_y_range[0]) * (expansion_factor - 1) / 2
                    new_y_range[0] = max(max_y_range[0], current_y_range[0] - expansion_size)
                    if verbose:
                        print(f"  Expanding y-axis lower bound")

                if at_y_max:
                    expansion_size = (current_y_range[1] - current_y_range[0]) * (expansion_factor - 1) / 2
                    new_y_range[1] = min(max_y_range[1], current_y_range[1] + expansion_size)
                    if verbose:
                        print(f"  Expanding y-axis upper bound")

                # Check if we've hit maximum ranges
                if (new_x_range[0] <= max_x_range[0] and new_x_range[1] >= max_x_range[1] and
                    new_y_range[0] <= max_y_range[0] and new_y_range[1] >= max_y_range[1]):
                    if verbose:
                        print(f"  Reached maximum search ranges")
                    break

                current_x_range = tuple(new_x_range)
                current_y_range = tuple(new_y_range)

        # Add search history to final result
        result.search_history = search_history

        if verbose:
            total_evals = sum(step['evaluations'] for step in search_history)
            print(f"\nSearch completed in {len(search_history)} expansions with {total_evals} total evaluations")
            print(f"Final best objective: {result.best_objective_value:.6f}")
            print(f"Final best coordinates: x={result.best_coordinates[0]:.6f}, y={result.best_coordinates[1]:.6f}")

        return result

    def plot_phase_plane_result(result: PhasePlaneResult,
                               title: str,
                               xlabel: str,
                               ylabel: str,
                               plot_type: str = '3d') -> None:
        """
        Plot phase plane results.

        Args:
            result: PhasePlaneResult object
            title, xlabel, ylabel: Plot labels
            plot_type: '2d', '3d', or 'both'
        """

        if plot_type in ['2d', 'both']:
            # 2D Contour Plot
            fig, ax = plt.subplots(figsize=(10, 8))

            contour = ax.contourf(result.X, result.Y, result.objective_matrix, 
                                 levels=20, cmap='viridis')
            contour_lines = ax.contour(result.X, result.Y, result.objective_matrix, 
                                      levels=10, colors='white', alpha=0.5, linewidths=0.5)
            ax.clabel(contour_lines, inline=True, fontsize=8)

            if not np.isnan(result.best_objective_value):
                ax.plot(result.best_coordinates[0], result.best_coordinates[1], 'r*', 
                       markersize=15, label=f'Best: {result.best_objective_value:.4f}')

            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label('Objective Value')

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{title}')
            if not np.isnan(result.best_objective_value):
                ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            if plot_type == '2d':
                plt.show()
                # uptake rates are negative of fluxes => put origin closest to observer
                ax.invert_xaxis()
                ax.invert_yaxis()

        if plot_type in ['3d', 'both']:
            # 3D Surface Plot
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection="3d")

            surf = ax.plot_surface(result.X, result.Y, result.objective_matrix, 
                                 cmap='viridis', alpha=0.8,
                                 linewidth=0, antialiased=True)

            ax.plot_wireframe(result.X, result.Y, result.objective_matrix, 
                            color='white', alpha=0.3, linewidth=0.5)

            if not np.isnan(result.best_objective_value):
                ax.scatter(result.best_coordinates[0], result.best_coordinates[1], 
                          result.best_objective_value, color='red', s=100, alpha=1.0,
                          label=f'Best: {result.best_objective_value:.4f}')

            cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
            cbar.set_label('Objective Value')

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel('Objective Value')
            ax.set_title(f'{title}')
            ax.view_init(elev=30, azim=45)

            if not np.isnan(result.best_objective_value):
                ax.legend()

            plt.tight_layout()

            if plot_type == '3d':
                plt.show()
                # uptake rates are negative of fluxes => put origin closest to observer
                ax.invert_xaxis()
                ax.invert_yaxis()

        if plot_type == 'both':
            plt.show()

    def plot_search_hierarchy(result: PhasePlaneResult, title: str = "Hierarchical Phase Plane Search"):
        """
        Visualize the hierarchical search process.

        Args:
            result: PhasePlaneResult with search_history
            title: Plot title
        """
        if result.search_history is None:
            print("No search history available for plotting")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Search regions at each level
        ax1.set_title("Search Regions by Level")
        colors = plt.cm.viridis(np.linspace(0, 1, len(result.search_history)))

        for i, step in enumerate(result.search_history):
            x_range = step['x_range']
            y_range = step['y_range']

            # Draw rectangle for search region
            rect = plt.Rectangle((x_range[0], y_range[0]), 
                               x_range[1] - x_range[0], 
                               y_range[1] - y_range[0],
                               fill=False, edgecolor=colors[i], linewidth=2,
                               label=f"Level {step['level']}")
            ax1.add_patch(rect)

            # Mark optimum for this level
            x_opt, y_opt = step['best_coordinates']
            ax1.plot(x_opt, y_opt, 'o', color=colors[i], markersize=8)

            # Add text annotation
            ax1.annotate(f"L{step['level']}", (x_opt, y_opt), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, color=colors[i], weight='bold')

        ax1.set_xlabel('X-axis Flux')
        ax1.set_ylabel('Y-axis Flux')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Objective value progression
        ax2.set_title("Best Objective by Level")
        levels = [step['level'] for step in result.search_history]
        objectives = [step['best_objective'] for step in result.search_history if not np.isnan(step['best_objective'])]
        evaluations = [step['evaluations'] for step in result.search_history]

        ax2_twin = ax2.twinx()

        line1 = ax2.plot(levels[:len(objectives)], objectives, 'bo-', linewidth=2, label='Best Objective')
        line2 = ax2_twin.plot(levels, evaluations, 'ro-', linewidth=2, label='Evaluations')

        ax2.set_xlabel('Refinement Level')
        ax2.set_ylabel('Best Objective Value', color='blue')
        ax2_twin.set_ylabel('Number of Evaluations', color='red')
        ax2.grid(True, alpha=0.3)

        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        plt.show()

    def analyze_phase_plane_statistics(result: PhasePlaneResult) -> dict:
        """
        Analyze statistics of phase plane results.

        Args:
            result: PhasePlaneResult object

        Returns:
            Dictionary containing various statistics
        """
        objective_matrix = result.objective_matrix
        X, Y = result.X, result.Y

        valid_mask = ~np.isnan(objective_matrix)
        valid_values = objective_matrix[valid_mask]

        if len(valid_values) == 0:
            return {"error": "No valid solutions found"}

        stats = {
            "mean_objective": np.mean(valid_values),
            "std_objective": np.std(valid_values),
            "min_objective": np.min(valid_values),
            "max_objective": np.max(valid_values),
            "feasible_fraction": np.sum(valid_mask) / objective_matrix.size,
            "total_points": objective_matrix.size,
            "feasible_points": np.sum(valid_mask),
            "x_range_final": result.x_range,
            "y_range_final": result.y_range,
            "total_evaluations": result.total_evaluations
        }

        # Find coordinates of best solutions (top 5%)
        if len(valid_values) > 0:
            threshold = np.percentile(valid_values, 95)
            top_solutions_mask = (objective_matrix >= threshold) & valid_mask
            top_x = X[top_solutions_mask]
            top_y = Y[top_solutions_mask]

            stats["top_5_percent_x_range"] = (np.min(top_x), np.max(top_x))
            stats["top_5_percent_y_range"] = (np.min(top_y), np.max(top_y))

        return stats
    return (
        adaptive_phase_plane_search,
        compute_phase_plane_grid,
        plot_phase_plane_result,
    )


@app.cell
def _():
    mo.md(r"""### CO2 vs Light Uptake""")
    return


@app.cell
def _(compute_phase_plane_grid, model, plot_phase_plane_result):
    PhPP_CO2_light = compute_phase_plane_grid(model,
                                              ["EX_CO2"],
                                              ["EX_PHO1", "EX_PHO2"],
                                              (-3200, 0),
                                              (-3200, 0),
                                              264,
                                              264)
    plot_phase_plane_result(PhPP_CO2_light,
                            title="$CO_2$ vs Light Uptake Phenotypic Phase Plane",
                            xlabel="$CO_2$ Exchange Flux (mmol/gDW/h)",
                            ylabel="Light Uptake: Photosystem I Uptake = Photosystem II Uptake ($\\mu E/m^2/s$)")
    return


@app.cell
def _(adaptive_phase_plane_search, model, plot_phase_plane_result):
    ### CO2 vs Ammonia
    adaptive_PhPP_CO2_light = adaptive_phase_plane_search(model,
                                               ["EX_CO2"],
                                               ["EX_PHO1", "EX_PHO2"],
                                               (-1200, 10),
                                               (-1200, 10),
                                               (-6000, 30),
                                               (-6000, 30))
    plot_phase_plane_result(adaptive_PhPP_CO2_light,
                            title="$CO_2$ vs Light Uptake Phenotypic Phase Plane",
                            xlabel="$CO_2$ Exchange Flux (mmol/gDW/h)",
                            ylabel="Light Uptake: Photosystem I Uptake = Photosystem II Uptake ($\\mu E/m^2/s$)")
    return (adaptive_PhPP_CO2_light,)


@app.cell
def _():
    mo.md(r"""Let's "lock in" these optimal CO2 and light uptakes.""")
    return


@app.cell
def _(adaptive_PhPP_CO2_light, model):
    model.EX_CO2.lower_bound, = adaptive_PhPP_CO2_light.best_coordinates[0]
    model.EX_PHO1.lower_bound = model.EX_PHO2.lower_bound = adaptive_PhPP_CO2_light.best_coordinates[1]
    return


@app.cell
def _():
    mo.md(r"""### CO2 vs Ammonia""")
    return


@app.cell
def _(adaptive_phase_plane_search, model, plot_phase_plane_result):
    PhPP_CO2_NH3 = adaptive_phase_plane_search(model,
                                               ["EX_CO2"],
                                               ["EX_NH3"],
                                               (-1200, 10),
                                               (-1200, 10),
                                               (-6000, 30),
                                               (-6000, 30))
    plot_phase_plane_result(PhPP_CO2_NH3,
                            title="$CO_2$ vs Ammonia Uptake Phenotypic Phase Plane",
                            xlabel="$CO_2$ Exchange Flux (mmol/gDW/h)",
                            ylabel="$NH_3$ Exchange Flux (mmol/gDW/h)")
    return


@app.cell
def _():
    mo.md(r"""### CO2 vs Nitrate""")
    return


@app.cell
def _(adaptive_phase_plane_search, model, plot_phase_plane_result):
    PhPP_CO2_Nitrate = adaptive_phase_plane_search(model,
                                               ["EX_CO2"],
                                               ["EX_Nitrate"],
                                               (-1200, 10),
                                               (-1200, 10),
                                               (-6000, 30),
                                               (-6000, 30))
    plot_phase_plane_result(PhPP_CO2_Nitrate,
                            title="$CO_2$ vs Nitrate Uptake Phenotypic Phase Plane",
                            xlabel="$CO_2$ Exchange Flux (mmol/gDW/h)",
                            ylabel="Nitrate Exchange Flux (mmol/gDW/h)")
    return


@app.cell
def _():
    mo.md(r"""### CO2 vs Phosphate""")
    return


@app.cell
def _(adaptive_phase_plane_search, model, plot_phase_plane_result):
    PhPP_CO2_Phosphate = adaptive_phase_plane_search(model,
                                               ["EX_CO2"],
                                               ["EX_Phosphate"],
                                               (-1200, 10),
                                               (-1200, 10),
                                               (-6000, 30),
                                               (-6000, 30))
    plot_phase_plane_result(PhPP_CO2_Phosphate,
                            title="$CO_2$ vs Phosphate Uptake Phenotypic Phase Plane",
                            xlabel="$CO_2$ Exchange Flux (mmol/gDW/h)",
                            ylabel="Phosphate Exchange Flux (mmol/gDW/h)")
    return


@app.cell
def _():
    mo.md(r"""### CO2 vs Citrate""")
    return


@app.cell
def _(adaptive_phase_plane_search, model, plot_phase_plane_result):
    PhPP_CO2_Citrate = adaptive_phase_plane_search(model,
                                               ["EX_CO2"],
                                               ["EX_Citrate"],
                                               (-1200, 10),
                                               (-1200, 10),
                                               (-6000, 30),
                                               (-6000, 30))
    plot_phase_plane_result(PhPP_CO2_Citrate,
                            title="$CO_2$ vs Citrate Uptake Phenotypic Phase Plane",
                            xlabel="$CO_2$ Exchange Flux (mmol/gDW/h)",
                            ylabel="Citrate Exchange Flux (mmol/gDW/h)")
    return


@app.cell
def _():
    mo.md(r"""### CO2 vs Sulfate""")
    return


@app.cell
def _(adaptive_phase_plane_search, model, plot_phase_plane_result):
    PhPP_CO2_Sulfate = adaptive_phase_plane_search(model,
                                               ["EX_CO2"],
                                               ["EX_Sulfate"],
                                               (-1200, 10),
                                               (-1200, 10),
                                               (-6000, 30),
                                               (-6000, 30))
    plot_phase_plane_result(PhPP_CO2_Sulfate,
                            title="$CO_2$ vs Sulfate Uptake Phenotypic Phase Plane",
                            xlabel="$CO_2$ Exchange Flux (mmol/gDW/h)",
                            ylabel="Sulfate Exchange Flux (mmol/gDW/h)")
    return


@app.cell
def _():
    mo.md(r"""### CO2 vs Calcium""")
    return


@app.cell
def _(adaptive_phase_plane_search, model, plot_phase_plane_result):
    PhPP_CO2_Calcium = adaptive_phase_plane_search(model,
                                               ["EX_CO2"],
                                               ["EX_Calcium"],
                                               (-1200, 10),
                                               (-1200, 10),
                                               (-6000, 30),
                                               (-6000, 30))
    plot_phase_plane_result(PhPP_CO2_Calcium,
                            title="$CO_2$ vs Calcium Uptake Phenotypic Phase Plane",
                            xlabel="$CO_2$ Exchange Flux (mmol/gDW/h)",
                            ylabel="Calcium Exchange Flux (mmol/gDW/h)")
    return


@app.cell
def _():
    mo.md(
        r"""
    ### CO2 vs Nitrate, Sulfate, Phosphate and Calcium
    CO2 and citrate should mainly be _carbon_ sources for UTEX. Then maybe for good growth, UTEX needs both abundant carbon supply and all these "supplements", like vitamins and proteins alongside main energy macronutrients--carbs and fats.
    """
    )
    return


@app.cell
def _(adaptive_phase_plane_search, model, plot_phase_plane_result):
    PhPP_CO2_multinutrients = adaptive_phase_plane_search(model,
                                                              ["EX_CO2"],
                                                              ["EX_Nitrate", "EX_Sulfate", "EX_Phosphate", "EX_Calcium"],
                                               (-1200, 10),
                                               (-1200, 10),
                                               (-6000, 30),
                                               (-6000, 30))
    plot_phase_plane_result(PhPP_CO2_multinutrients,
                            title="$CO_2$ vs Nitrate, Sulfate, Phosphate and Calcium Uptake Phenotypic Phase Plane",
                            xlabel="$CO_2$ Exchange Flux (mmol/gDW/h)",
                            ylabel="Nitrate = Sulfate = Phosphate = Calcium Exchange Flux (mmol/gDW/h)")
    return


@app.cell
def _():
    mo.md(r"""## Flux Variability Analysis on BG-11 Components Uptakes""")
    return


@app.cell
def _():
    mo.md(r"""Extend the default flux constraints in the model so we actually explore enough of the space of possibilities.""")
    return


app._unparsable_cell(
    r"""
    import pandas as pd
    from cobra.flux_analysis import flux_variability_analysis

    def run_fva_with_lbs(model: cobra.core.Model, lower_bounds: dict[str, float]):
        \"\"\"Run flux variability analysis on the biomass reaction with specified lower bounds on specified exchange reactions as a dict of reaction IDs to lower bounds.\"\"\"
        with model:
            for _rxn_id, _lb in lower_bounds.items():
                if _rxn_id in model.reactions:
                    _rxn = model.reactions.get_by_id(_rxn_id)
                    _rxn.lower_bound = _lb
            return flux_variability_analysis(model, [biomass_rxn], fraction_of_optimum=1, loopless=True)])

    def plot_fva_ranges(fva_results: pd.DataFrame, title: str):
        \"\"\"Plot horizontal bar plot of FVA results.\"\"\"

        plt.figure(figsize=(6, 10))
        ax = plt.gca()
        bar_plot = plt.barh(fva_results.index, fva_results['maximum'] - fva_results['minimum'], left=fva_results['minimum'], color='violet')
        for (i, (min_val, max_val)) in enumerate(zip(fva_results['minimum'], fva_results['maximum'])):
            ax.plot(min_val, i, marker='|', color='black', markersize=18, markeredgewidth=2)
            ax.plot(max_val, i, marker='|', color='black', markersize=18, markeredgewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Flux Range (mmol/gDW/h)')
        ax.set_ylabel('Reactions')
        plt.tight_layout()
        plt.show()
    """,
    name="_"
)


@app.cell
def _():
    mo.md(
        r"""
    From the above phenotypic phase plane analyses, we know optimal...

    - CO2 = -132 mmol/gDW/h, photosystem I = photosystem II = -900 mmol/gDW/h
    - CO2 = -30 mmol/gDW/h, NH3 = -30 mmol/gDW/h

    Let's go to...

    - -2000 $\leq$ CO2
    - -2500 $\leq$ photosystems
    - -1000 $\leq$ others
    """
    )
    return


@app.cell
def _(BG_11_MEDIA_EXRXNS_IDS, PROCD_DATA_DIR, model, pd):
    PhPP_lbs = {_rxn_id : min(-1500, model.reactions.get_by_id(_rxn_id).lower_bound) for _rxn_id in BG_11_MEDIA_EXRXNS_IDS}
    PhPP_lbs["EX_CO2"] = -2000
    PhPP_lbs["EX_PHO1"] = -2500
    PhPP_lbs["EX_PHO2"] = -2500

    PhPP_lbs_series = pd.Series(PhPP_lbs)
    print(PhPP_lbs_series)
    PhPP_lbs_series.to_csv(PROCD_DATA_DIR / "BG11_PhPP_lower_bounds0.csv")
    return (PhPP_lbs,)


@app.cell
def _(PhPP_lbs, model, run_fva_with_lbs):
    fva_results_PhPP = run_fva_with_lbs(model, PhPP_lbs)
    fva_results_PhPP
    return (fva_results_PhPP,)


@app.cell
def _(fva_results_PhPP, plot_fva_ranges):
    plot_fva_ranges(fva_results_PhPP, title="FVA of BG-11 Media Uptake Reactions with Phase Plane Analysis Optimal Lower Bounds")
    return


@app.cell
def _():
    mo.md(
        r"""
    ### _Note_: Beware Loops
    The [COBRApy docs](https://cobrapy.readthedocs.io/en/latest/simulating.html#Running-FVA) note that unrealistic **loops** of reactions may be simulated, leading to some **unrealistically high fluxes**. Luckily, COBRApy includes detecting such loops and ensuring solutions without their resulting artifacts.

    Let's see if we got any such loops here.
    """
    )
    return


@app.cell
def _(BG_11_MEDIA_EXRXNS_IDS, flux_variability_analysis, model):
    loopless_fva_results = flux_variability_analysis(model, [model.reactions.get_by_id(_rxn_id) for _rxn_id in BG_11_MEDIA_EXRXNS_IDS], fraction_of_optimum=1, loopless=True)
    loopless_fva_results
    return (loopless_fva_results,)


@app.cell
def _(PhPP_fva_results, loopless_fva_results):
    # check the difference between loopless and non-loopless FVA
    PhPP_fva_results - loopless_fva_results
    return


@app.cell
def _():
    mo.md(
        r"""
    ### No Longer Using CO2 as Main Carbon Source?
    Ideally, we would want cyano to use CO2 as citrate, carbonic acid, etc. would need to be transported to Mars.
    """
    )
    return


app._unparsable_cell(
    r"""
    EXTEND_EXRXNS_IDS = {'EX_CO2', 'EX_PHO1', 'EX_PHO2', 'EX_NH3', 'EX_Mn2_', 'EX_Zinc', 'EX_Cu2_', 'EX_Molybdate', 'EX_Co2_', 'EX_Nitrate', 'EX_Phosphate', 'EX_Sulfate', 'EX_Fe3_', 'EX_Calcium'}
    PhPP_extended_lbs = {_rxn_id : min(-1500, model.reactions.get_by_id(_rxn_id).lower_bound) for _rxn_id in EXTEND_EXRXNS_IDS}
    PhPP_extended_lbs[EX_CO2\"].lower_bound = -2000
    PhPP_extended_lbs[EX_PHO1\"].lower_bound = -2500
    PhPP_extended_lbs[EX_PHO2\"].lower_bound = -2500
    PhPP_extended_lbs[EX_Citrate\"].lower_bound = -1
    PhPP_extended_lbs[EX_H2CO3\"].lower_bound = -10

    pd.Series(PhPP_extended_lbs).to_csv(PROCD_DATA_DIR / \"BG11_exchange_lower_bounds1.csv\")

    fva_results_PhPP_bounds_extended = run_fva_with_lbs(model, PhPP_extended_lbs)
    """,
    name="_"
)


@app.cell
def _(fva_results_PhPP_bounds_extended):
    fva_results_PhPP_bounds_extended
    return


@app.cell
def _(fva_results_PhPP, plot_fva_ranges):
    plot_fva_ranges(fva_results_PhPP, title="FVA of BG-11 Media Uptake Reactions with Lower Bounds Beyond Phase Plane Analysis Optimal Values")
    return


@app.cell
def _():
    mo.md(
        r"""
    ## Some Proper Optimization
    Ok, what we have here is an optimization over a moderately high-dimensional parameter space with quite a tractable objective function (FBA solve = linear programming). It's time to use a framework like [Ax](https://ax.dev/).
    """
    )
    return


@app.cell
def _(model):
    def fba_solve(model, flux_bounds: dict[str, tuple[float, float]]):
        for _rxn_id, (_lower_bound, _upper_bound) in flux_bounds.items():
            if _rxn_id in model.reactions:
                _rxn = model.reactions.get_by_id(_rxn_id)
                _rxn.lower_bound = _lower_bound
                _rxn.upper_bound = _upper_bound
        return model.slim_optimize()

    def BG11_uptakes_objective(lb_EX_CO2: float,
                                lb_EX_PHO1: float,
                                lb_EX_PHO2: float,
                                lb_EX_NH3: float,
                                lb_EX_Mn2_: float,
                                lb_EX_Zinc: float,
                                lb_EX_Cu2_: float,
                                lb_EX_Molybdate: float,
                                lb_EX_Co2_: float,
                                lb_EX_Nitrate: float,
                                lb_EX_Phosphate: float,
                                lb_EX_Sulfate: float,
                                lb_EX_Fe3_: float,
                                lb_EX_Calcium: float,
                                lb_EX_Citrate: float,
                                lb_EX_H2CO3: float) -> float:
        flux_bounds = {
            "EX_CO2": (lb_EX_CO2, 0),
            "EX_PHO1": (lb_EX_PHO1, 0),
            "EX_PHO2": (lb_EX_PHO2, 0),
            "EX_NH3": (lb_EX_NH3, 0),
            "EX_Mn2_": (lb_EX_Mn2_, 0),
            "EX_Zinc": (lb_EX_Zinc, 0),
            "EX_Cu2_": (lb_EX_Cu2_, 0),
            "EX_Molybdate": (lb_EX_Molybdate, 0),
            "EX_Co2_": (lb_EX_Co2_, 0),
            "EX_Nitrate": (lb_EX_Nitrate, 0),
            "EX_Phosphate": (lb_EX_Phosphate, 0),
            "EX_Sulfate": (lb_EX_Sulfate, 0),
            "EX_Fe3_": (lb_EX_Fe3_, 0),
            "EX_Calcium": (lb_EX_Calcium, 0),
            "EX_Citrate": (lb_EX_Citrate, 0),
            "EX_H2CO3": (lb_EX_H2CO3, 0)
        }
        with model:
            return fba_solve(model, flux_bounds)
    return (BG11_uptakes_objective,)


@app.cell
def _():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    return (device,)


@app.cell
def _(device):
    params = [
        RangeParameterConfig(name="lb_EX_CO2", parameter_type="float", bounds=(-3600, -1000)),
        RangeParameterConfig(name="lb_EX_PHO1", parameter_type="float", bounds=(-3000, -1000)),
        RangeParameterConfig(name="lb_EX_PHO2", parameter_type="float", bounds=(-3000, -1000)),
        RangeParameterConfig(name="lb_EX_NH3", parameter_type="float", bounds=(-3000, -50)),
        RangeParameterConfig(name="lb_EX_Mn2_", parameter_type="float", bounds=(-1000, 0)),
        RangeParameterConfig(name="lb_EX_Zinc", parameter_type="float", bounds=(-1000, 0)),
        RangeParameterConfig(name="lb_EX_Cu2_", parameter_type="float", bounds=(-1000, 0)),
        RangeParameterConfig(name="lb_EX_Molybdate", parameter_type="float", bounds=(-1000, 0)),
        RangeParameterConfig(name="lb_EX_Co2_", parameter_type="float", bounds=(-1000, 0)),
        RangeParameterConfig(name="lb_EX_Nitrate", parameter_type="float", bounds=(-1000, -50)),
        RangeParameterConfig(name="lb_EX_Phosphate", parameter_type="float", bounds=(-1000, -10)),
        RangeParameterConfig(name="lb_EX_Sulfate", parameter_type="float", bounds=(-1000, -10)),
        RangeParameterConfig(name="lb_EX_Fe3_", parameter_type="float", bounds=(-1000, 0)),
        RangeParameterConfig(name="lb_EX_Calcium", parameter_type="float", bounds=(-1000, 0)),
        RangeParameterConfig(name="lb_EX_Citrate", parameter_type="float", bounds=(-3000, 0)),
        RangeParameterConfig(name="lb_EX_H2CO3", parameter_type="float", bounds=(-500, 0))
    ]

    client = Client()
    client.configure_experiment(parameters=params)
    client.configure_optimization(objective="BG11_uptakes_objective")
    client.configure_generation_strategy(allow_exceeding_initialization_budget=True, torch_device=device)
    return (client,)


@app.cell
def _():
    mo.md(r"""Now the optimization is finally set up, and the optimization loop can actually begin.""")
    return


@app.cell
def _(BG11_uptakes_objective, client):
    NUM_ROUNDS_OPTIM = 20

    for i_opt_round in range(NUM_ROUNDS_OPTIM):
        trials = client.get_next_trials(max_trials=2)

        for i_trial, parameters in trials.items():
            print(f"Trial {i_trial}: {parameters}")

            objective_value = BG11_uptakes_objective(**parameters)
            raw_data = {"BG11_uptakes_objective": objective_value}

            client.complete_trial(trial_index=i_trial, raw_data=raw_data)
    return


@app.cell
def _(client):
    best_params, pred, best_idx, nm = client.get_best_parameterization()
    print("Best parameters:", best_params)
    print(f"Best objective value: {pred[0]} +/- {pred[1]}")
    return


@app.cell
def _(client):
    cards = client.compute_analyses(display=True)
    return


@app.cell
def _():
    mo.md(
        r"""
    # Translating to Real Concentrations
    Now that we finally have the predicted best set of uptake rates for BG-11 media components, it's time to put them to actual use in the wet lab.

    So, how much of each component do we add to our media? We need to translate *uptake rates* to *concentrations*. *Fluxes* are **not** equivalent to *concentrations*--the former are in $\mathrm{mmol} \cdot \mathrm{gDW^{-1}} \cdot \mathrm{hr^{-1}}$, while the latter in $\mathrm{g} \cdot \mathrm{L^{-1}}$. It's not just a simple unit conversion. Instead, we've a bit more planning to do and assumptions to make.

    - For now, let's say we'll only "feed" our cyano culture once, at the start of the experiment. So we need to make sure that the initial concentration is enough to sustain the uptake for the entire duration of the experiment.
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    Now for translating fluxes to concentrations, we need to compute compute projected growth to multiply away the $\mathrm{gDW}$ in the denominator. We can get this from
    $\frac {m_0 + \Delta m} {m_0}
    = \frac
        {m_0 + \int_{t=0}^{\Delta t} v_{\mathtt{biomass}} \mathrm{d}t}
        {m_0}
    = \frac
        {m_0 + v_{\mathtt{biomass}}\Delta t}
        {m_0}$,
    except that $v_{\mathtt{biomass}}$ is in $\mathrm{mmol} \cdot \mathrm{gDW}^{-1} \cdot \mathrm{hr}^{-1}$, so $v_{\mathtt{biomass}}\Delta t$ would be in $\mathrm{mmol} \cdot \mathrm{gDW}^{-1}$.

    Now, remember, $\mathrm{mmol}$ of what? $\text{mmol biomass}$. And what is dry weight? _Biomass_. Then all that's left is to figure out how many grams of biomass is one mmol of biomass, which means we need to find the **molecular weight of biomass** as defined by the biomass exchange "reaction".
    """
    )
    return


@app.cell
def _(model):
    # Let's dig up the composition of the biomass objective function
    model
    return


@app.cell
def _(model):
    model.objective
    return


@app.cell
def _(model):
    biomass_rxn = model.reactions.Biomass_Auto_2973
    biomass_rxn
    return (biomass_rxn,)


@app.cell
def _(biomass_rxn):
    biomass_rxn.check_mass_balance()
    return


@app.cell
def _():
    mo.md(r"""Uhhhhhmmm... well that's not good.""")
    return


@app.cell
def _(biomass_rxn):
    biomass_rxn.products
    return


@app.cell
def _():
    mo.md(r"""Let's check if `Metabolite`s' `formula_weight`s are molecular masses. For example, ADP's molar mass should be 427.201 g/mol.""")
    return


@app.cell
def _(model):
    model.metabolites.get_by_id("cpd00008_c").formula_weight
    return


@app.cell
def _(model):
    biomass_metabolite = model.metabolites.get_by_id("cpd11416_c")
    biomass_metabolite.formula_weight
    return


if __name__ == "__main__":
    app.run()
