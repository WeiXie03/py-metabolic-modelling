import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
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
def _(mo):
    mo.md(r"""## Load model""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
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
def _(mo):
    mo.md(r"""_Note_: The below cell might take a bit long to run if the SBML model is big.""")
    return


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    import cobra
    from cobra.io import read_sbml_model, save_json_model
    from cobra import Model, Reaction, Metabolite

    RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
    PROCD_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
    # RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
    # PROCD_DATA_DIR = ROOT_DIR / "data" / "processed"

    model = read_sbml_model(RAW_DATA_DIR / "SBMLmodel_UTEX2973.xml")
    return cobra, mo, model


@app.cell
def _(mo):
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
def _(mo):
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
def _(mo):
    mo.md(
        r"""
    ## Phenotypic Phase Planes
    First explore some potentially important components, varying pairs simultaneously at a time.
    """
    )
    return


@app.cell
def _(cobra, plt):
    import numpy as np
    from typing import List, Tuple, Optional

    def plot_phenotypic_phase_plane(model: cobra.Model,
                                   rxnsIDs_x: List[str],
                                   rxnsIDs_y: List[str],
                                   x_range_spec: Tuple[float, float, int],
                                   y_range_spec: Tuple[float, float, int],
                                   title: str,
                                   xlabel: str,
                                   ylabel: str,
                                   obj_rxn_id: Optional[str] = None,
                                   plot: bool = True) -> Tuple[Tuple[float, float], float, np.ndarray]:
        """
        Plots the phenotypic phase plane for two sets of reactions in a COBRApy model.

        Args:
            model (cobra.Model): The COBRApy model to analyze.
            rxnsIDs_x: List of IDs of reactions for the x-axis. If > 1 reaction, 
                       at every step all their flux lower bounds will be set to the same step value.
            rxnsIDs_y: List of IDs of reactions for the y-axis. If > 1 reaction, 
                       at every step all their flux lower bounds will be set to the same step value.
            x_range_spec: Tuple (min, max, num_steps) specifying the range and number of steps for the x-axis reactions.
            y_range_spec: Tuple (min, max, num_steps) specifying the range and number of steps for the y-axis reactions.
            title: Title of the plot.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            obj_rxn_id: ID of the reaction to optimize (default: model's current objective).
            plot: Whether to create the plot (default: True).

        Returns:
            tuple: (best_lbs, best_objective_value, objective_matrix) where:
                   - best_lbs is a tuple of the best lower bounds (x, y)
                   - best_objective_value is the corresponding objective value
                   - objective_matrix is the 2D array of objective values for plotting
        """

        # Input validation
        for rxn_id in rxnsIDs_x + rxnsIDs_y:
            if rxn_id not in model.reactions:
                raise ValueError(f'Reaction {rxn_id} not found in the model.')

        with model:
            # Set objective if specified
            if obj_rxn_id is not None:
                model.objective = obj_rxn_id

            # Create coordinate arrays using numpy
            x_values = np.linspace(x_range_spec[0], x_range_spec[1], x_range_spec[2])
            y_values = np.linspace(y_range_spec[0], y_range_spec[1], y_range_spec[2])

            # Create meshgrid for coordinates
            X, Y = np.meshgrid(x_values, y_values, indexing='ij')

            # Initialize results matrix, biomass accumulation rate is a flux => >= 0
            objective_matrix = np.zeros(X.shape)

            # Iterate through the meshgrid using numpy's flat indexing
            for i, (x_val, y_val) in enumerate(zip(X.flat, Y.flat)):
                # Convert flat index back to 2D indices
                idx_2d = np.unravel_index(i, X.shape)

                # Set lower bounds for x-axis reactions
                for rxn_id in rxnsIDs_x:
                    model.reactions.get_by_id(rxn_id).lower_bound = x_val
                # Set lower bounds for y-axis reactions  
                for rxn_id in rxnsIDs_y:
                    model.reactions.get_by_id(rxn_id).lower_bound = y_val

                # Optimize and store result
                solution = model.slim_optimize(error_value= np.nan, message = f"Optimization failed at x={x_val:.3f}, y={y_val:.3f}")
                # If optimization fails, slim_optimize will return NaN
                objective_matrix[idx_2d] = solution

        # Find best solution using numpy operations
        # Handle NaN values by creating a mask
        valid_mask = ~np.isnan(objective_matrix)

        if not np.any(valid_mask):
            print("No feasible solutions found in the specified ranges.")
            return (None, None), np.nan, objective_matrix

        # Find the maximum value and its location
        best_idx = np.nanargmax(objective_matrix)
        best_idx_2d = np.unravel_index(best_idx, objective_matrix.shape)
        best_objective_value = objective_matrix[best_idx_2d]
        best_lbs = (X[best_idx_2d], Y[best_idx_2d])

        print(f'Best objective value {best_objective_value:.6f} found at:')
        print(f'  {rxnsIDs_x} lower bounds = {best_lbs[0]:.6f}')
        print(f'  {rxnsIDs_y} lower bounds = {best_lbs[1]:.6f}')

        # Create the plot if requested
        if plot:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create contour plot
            contour = ax.contourf(X, Y, objective_matrix, levels=20, cmap='viridis')

            # Add contour lines
            contour_lines = ax.contour(X, Y, objective_matrix, levels=10, colors='white', alpha=0.5, linewidths=0.5)
            ax.clabel(contour_lines, inline=True, fontsize=8)

            # Mark the best point
            if not np.isnan(best_objective_value):
                ax.plot(best_lbs[0], best_lbs[1], 'r*', markersize=15, label=f'Best: {best_objective_value:.4f}')

            # Add colorbar
            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label('Objective Value')

            # Labels and title
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        return best_lbs, best_objective_value, objective_matrix


    def analyze_phase_plane_statistics(objective_matrix: np.ndarray, 
                                     X: np.ndarray, 
                                     Y: np.ndarray) -> dict:
        """
        Analyze statistics of the phase plane results.

        Args:
            objective_matrix: 2D array of objective values
            X, Y: Meshgrid arrays for x and y coordinates

        Returns:
            Dictionary containing various statistics
        """
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
            "feasible_points": np.sum(valid_mask)
        }

        # Find coordinates of best solutions (top 5%)
        threshold = np.percentile(valid_values, 95)
        top_solutions_mask = (objective_matrix >= threshold) & valid_mask
        top_x = X[top_solutions_mask]
        top_y = Y[top_solutions_mask]

        stats["top_5_percent_x_range"] = (np.min(top_x), np.max(top_x))
        stats["top_5_percent_y_range"] = (np.min(top_y), np.max(top_y))

        return stats
    return (plot_phenotypic_phase_plane,)


@app.cell
def _(mo):
    mo.md(r"""### CO2 vs Light Uptake""")
    return


@app.cell
def _(model, plot_phenotypic_phase_plane):
    plot_phenotypic_phase_plane(model,
                                ["EX_CO2"],
                                ["EX_PHO1", "EX_PHO2"],
                                (-3200, 0, 264),
                                (-3200, 0, 264),
                                title="CO_2 vs Light Uptake Phenotypic Phase Plane",
                                xlabel="CO_2 Uptake (mmol/gDW/h)",
                                ylabel="Light Uptake: Photosystem I Uptake = Photosystem II Uptake (\\mu E/m^2/s)")
    return


@app.cell
def _(mo):
    mo.md(r"""### CO2 vs Ammonia""")
    return


@app.cell
def _(model, plot_phenotypic_phase_plane):
    plot_phenotypic_phase_plane(model,
                                ["EX_CO2"],
                                ["EX_NH3"],
                                (-3200, 0, 264),
                                (-3200, 0, 264),
                                title="CO2 vs Ammonia Uptake Phenotypic Phase Plane",
                                xlabel="CO_2 Uptake (mmol/gDW/h)",
                                ylabel="NH_3 Uptake (mmol/gDW/h)")
    return


@app.cell
def _(mo):
    mo.md(r"""### CO2 vs Nitrate""")
    return


@app.cell
def _(model, plot_phenotypic_phase_plane):
    plot_phenotypic_phase_plane(model,
                                ["EX_CO2"],
                                ["EX_Nitrate"],
                                (-3200, 0, 264),
                                (-3200, 0, 264),
                                title="CO_2 vs Nitrate Uptake Phenotypic Phase Plane",
                                xlabel="CO_2 Uptake (mmol/gDW/h)",
                                ylabel="Nitrate Uptake (mmol/gDW/h)")
    return


@app.cell
def _(mo):
    mo.md(r"""### CO2 vs Phosphate""")
    return


@app.cell
def _(model, plot_phenotypic_phase_plane):
    plot_phenotypic_phase_plane(model,
                                ["EX_CO2"],
                                ["EX_Phosphate"],
                                (-1000, 0, 100),
                                (-1000, 0, 100),
                                title="CO_2 vs Phosphate Uptake Phenotypic Phase Plane",
                                xlabel="CO_2 Uptake (mmol/gDW/h)",
                                ylabel="Phosphate Uptake (mmol/gDW/h)")
    return


@app.cell
def _(mo):
    mo.md(r"""### CO2 vs Citrate""")
    return


@app.cell
def _(model, plot_phenotypic_phase_plane):
    plot_phenotypic_phase_plane(model,
                                ["EX_CO2"],
                                ["EX_Citrate"],
                                (-1000, 0, 100),
                                (-1000, 0, 100),
                                title="CO_2 vs Citrate Uptake Phenotypic Phase Plane",
                                xlabel="CO_2 Uptake (mmol/gDW/h)",
                                ylabel="Citrate Uptake (mmol/gDW/h)")
    return


@app.cell
def _(mo):
    mo.md(r"""### CO2 vs Sulfate""")
    return


@app.cell
def _(model, plot_phenotypic_phase_plane):
    plot_phenotypic_phase_plane(model,
                                ["EX_CO2"],
                                ["EX_Sulfate"],
                                (-1000, 0, 100),
                                (-1000, 0, 100),
                                title="CO_2 vs Sulfate Uptake Phenotypic Phase Plane",
                                xlabel="CO_2 Uptake (mmol/gDW/h)",
                                ylabel="Sulfate Uptake (mmol/gDW/h)")
    return


@app.cell
def _(mo):
    mo.md(r"""### CO2 vs Calcium""")
    return


@app.cell
def _(model, plot_phenotypic_phase_plane):
    plot_phenotypic_phase_plane(model,
                                ["EX_CO2"],
                                ["EX_Calcium"],
                                (-1000, 0, 100),
                                (-1000, 0, 100),
                                title="CO_2 vs Calcium Uptake Phenotypic Phase Plane",
                                xlabel="CO_2 Uptake (mmol/gDW/h)",
                                ylabel="Calcium Uptake (mmol/gDW/h)")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### CO2 vs Nitrate, Sulfate, Phosphate and Calcium
    CO2 and citrate should mainly be _carbon_ sources for UTEX. Then maybe for good growth, UTEX needs both abundant carbon supply and all these "supplements", like vitamins and proteins alongside main energy macronutrients--carbs and fats.
    """
    )
    return


@app.cell
def _(model, plot_phenotypic_phase_plane):
    plot_phenotypic_phase_plane(model,
                                ["EX_CO2"],
                                ["EX_Nitrate", "EX_Sulfate", "EX_Phosphate", "EX_Calcium"],
                                (-1000, 0, 200),
                                (-1000, 0, 200),
                                title="CO_2 vs Nitrate, Sulfate, Phosphate and Calcium Uptake Phenotypic Phase Plane",
                                xlabel="CO_2 Uptake (mmol/gDW/h)",
                                ylabel="Nitrate = Sulfate = Phosphate = Calcium Uptake (mmol/gDW/h)")
    return


@app.cell
def _(mo):
    mo.md(r"""## Flux Variability Analysis on BG-11 Components Uptakes""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Extend the default flux constraints in the model so we actually explore enough of the space of possibilities.

    From the above (_TODO_) phenotypic phase plane analyses, we know optimal...
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
def _(BG_11_MEDIA_EXRXNS_IDS, model):
    model_extended_constraints = model.copy()
    for _rxn_id in BG_11_MEDIA_EXRXNS_IDS:
        _rxn = model_extended_constraints.reactions.get_by_id(_rxn_id)
        if _rxn.lower_bound > -1500:
            _rxn.lower_bound = -1500
    model_extended_constraints.reactions.EX_CO2.lower_bound = -2000
    model_extended_constraints.reactions.EX_PHO1.lower_bound = -2500
    model_extended_constraints.reactions.EX_PHO2.lower_bound = -2500
    for _rxn_id in BG_11_MEDIA_EXRXNS_IDS:
        _rxn = model_extended_constraints.reactions.get_by_id(_rxn_id)
        print(f'{_rxn.id}: {_rxn.lower_bound} <= flux <= {_rxn.upper_bound}')
    return (model_extended_constraints,)


@app.cell
def _(BG_11_MEDIA_EXRXNS_IDS, model_extended_constraints):
    from cobra.flux_analysis import flux_variability_analysis
    fva_results = flux_variability_analysis(model_extended_constraints, [model_extended_constraints.reactions.get_by_id(_rxn_id) for _rxn_id in BG_11_MEDIA_EXRXNS_IDS], fraction_of_optimum=1, loopless=True)
    fva_results
    return flux_variability_analysis, fva_results


@app.cell
def _(fva_results):
    from matplotlib import pyplot as plt
    plt.figure(figsize=(6, 10))
    _ax = plt.gca()
    _bar_plot = plt.barh(fva_results.index, fva_results['maximum'] - fva_results['minimum'], left=fva_results['minimum'], color='violet')
    for (_i, (_min_val, _max_val)) in enumerate(zip(fva_results['minimum'], fva_results['maximum'])):
        _ax.plot(_min_val, _i, marker='|', color='black', markersize=18, markeredgewidth=2)
        _ax.plot(_max_val, _i, marker='|', color='black', markersize=18, markeredgewidth=2)
    _ax.set_title('Flux Variability Analysis of BG-11 Media Uptake Reactions')
    _ax.set_xlabel('Flux Range (mmol/gDW/h)')
    _ax.set_ylabel('Reactions')
    plt.tight_layout()
    plt.show()
    return (plt,)


@app.cell
def _(mo):
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
def _(fva_results, loopless_fva_results):
    # check the difference between loopless and non-loopless FVA
    fva_results - loopless_fva_results
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### No Longer Using CO2 as Main Carbon Source?
    Ideally, we would want cyano to use CO2 as citrate, carbonic acid, etc. would need to be transported to Mars.
    """
    )
    return


@app.cell
def _(BG_11_MEDIA_EXRXNS_IDS, model):
    model_extended_constraints_1 = model.copy()
    EXTEND_EXRXNS_IDS = {'EX_CO2', 'EX_PHO1', 'EX_PHO2', 'EX_NH3', 'EX_Mn2_', 'EX_Zinc', 'EX_Cu2_', 'EX_Molybdate', 'EX_Co2_', 'EX_Nitrate', 'EX_Phosphate', 'EX_Sulfate', 'EX_Fe3_', 'EX_Calcium'}
    for _rxn_id in EXTEND_EXRXNS_IDS:
        _rxn = model_extended_constraints_1.reactions.get_by_id(_rxn_id)
        if _rxn.lower_bound > -1500:
            _rxn.lower_bound = -1500
    model_extended_constraints_1.reactions.EX_CO2.lower_bound = -2000
    model_extended_constraints_1.reactions.EX_PHO1.lower_bound = -2500
    model_extended_constraints_1.reactions.EX_PHO2.lower_bound = -2500
    model_extended_constraints_1.reactions.EX_Citrate.lower_bound = -1
    model_extended_constraints_1.reactions.EX_H2CO3.lower_bound = -10
    for _rxn_id in BG_11_MEDIA_EXRXNS_IDS:
        _rxn = model_extended_constraints_1.reactions.get_by_id(_rxn_id)
        print(f'{_rxn.id}: {_rxn.lower_bound} <= flux <= {_rxn.upper_bound}')
    return (model_extended_constraints_1,)


@app.cell
def _(
    BG_11_MEDIA_EXRXNS_IDS,
    flux_variability_analysis,
    model_extended_constraints_1,
):
    fva_results_1 = flux_variability_analysis(model_extended_constraints_1, [model_extended_constraints_1.reactions.get_by_id(_rxn_id) for _rxn_id in BG_11_MEDIA_EXRXNS_IDS], fraction_of_optimum=1, loopless=True)
    fva_results_1
    return (fva_results_1,)


@app.cell
def _(fva_results_1, plt):
    plt.figure(figsize=(6, 10))
    _ax = plt.gca()
    _bar_plot = plt.barh(fva_results_1.index, fva_results_1['minimum'] - fva_results_1['maximum'], left=-fva_results_1['minimum'], color='violet')
    for (_i, (_min_val, _max_val)) in enumerate(zip(-fva_results_1['maximum'], -fva_results_1['minimum'])):
        _ax.plot(_min_val, _i, marker='|', color='black', markersize=18, markeredgewidth=2)
        _ax.plot(_max_val, _i, marker='|', color='black', markersize=18, markeredgewidth=2)
    _ax.set_title('Flux Variability Analysis of BG-11 Media Uptake Reactions')
    _ax.set_xlabel('Uptake Range (mmol/gDW/h)')
    _ax.set_ylabel('Reactions')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
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
        solution = model.optimize()
        return solution.objective_value

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
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    return (device,)


@app.cell
def _(device):
    from ax import Client, RangeParameterConfig

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
def _(mo):
    mo.md(r"""Now the optimization is finally set up, and the optimization loop can actually begin.""")
    return


@app.cell(disabled=True)
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
    # best_params, pred, best_idx, nm = client.get_best_parameterization()
    client.get_best_parameterization()
    # print("Best parameters:", best_params)
    # print("Best objective value: (mean, variance)", pred)
    return


@app.cell
def _(client):
    cards = client.compute_analyses(display=True)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
