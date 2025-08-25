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
    return mo, model


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
        RangeParameterConfig(name="lb_EX_Nitrate", parameter_type="float", bounds=(-3200, -50)),
        RangeParameterConfig(name="lb_EX_Phosphate", parameter_type="float", bounds=(-3200, -10)),
        RangeParameterConfig(name="lb_EX_Sulfate", parameter_type="float", bounds=(-3200, -10)),
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
