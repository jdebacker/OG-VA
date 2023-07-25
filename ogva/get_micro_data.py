"""
-------------------------------------------------------------------------------
This program extracts tax rate and income data from the microsimulation model
model (FiscalSim-US)
-------------------------------------------------------------------------------
"""
from dask import delayed, compute
import dask.multiprocessing
import numpy as np
import os
import pickle
from fiscalsim_us import Microsimulation
import pandas as pd
import warnings
from fiscalsim_us import *
from policyengine_core.reforms import Reform
import logging

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore")

CUR_PATH = os.path.split(os.path.abspath(__file__))[0]
DATA_LAST_YEAR = 2023  # this is the last year data are extrapolated for


def get_household_mtrs(
    reform,
    variable: str,
    period: int = None,
    **kwargs: dict,
) -> pd.Series:
    """Calculates household MTRs with respect to a given variable.

    Args:
        reform (ReformType): The reform to apply to the simulation.
        variable (str): The variable to increase.
        period (int): The period (year) to calculate the MTRs for.
        kwargs (dict): Additional arguments to pass to the simulation.

    Returns:
        pd.Series: The household MTRs.
    """
    baseline = Microsimulation(reform=reform, **kwargs)
    baseline_var = baseline.calc(variable, period)
    bonus = baseline.calc("is_adult", period) * 1  # Increase only adult values
    reformed = Microsimulation(reform=reform, **kwargs)
    reformed.set_input(variable, period, baseline_var + bonus)

    household_bonus = reformed.calc(
        variable, map_to="household", period=period
    ) - baseline.calc(variable, map_to="household", period=period)
    household_net_change = reformed.calc(
        "household_net_income", period=period
    ) - baseline.calc("household_net_income", period=period)
    mtr = (household_bonus - household_net_change) / household_bonus
    mtr = mtr.replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 0.99)
    return mtr


def get_calculator_output(baseline, year, reform=None, data=None):
    """
    This function creates a FiscalSim Microsimulation object with the policy
    specified in reform and the data specified with the data kwarg.

    Args:
        baseline (boolean): True if baseline tax policy
        year (int): year of data to simulate
        reform (PolicyEngine Reform object): IIT policy reform parameters,
            None if baseline
        data (DataFrame or str): DataFrame or path to datafile for
            the PopulationSim object

    Returns:
        tax_dict (dict): a dictionary of microdata with marginal tax
            rates and other information computed from PolicyEngine-UK

    """
    # create a simulation
    # sim_kwargs = dict(dataset=data, dataset_year=2023)
    sim_kwargs = {}
    # def modify_parameters_rfm(parameters):
    #     """
    #     Baseline reform is to not modify the parameters.
    #     """
    #     pass
    #     return parameters

    # class cls_reform(Reform):
    #     def apply(self):
    #         self.modify_parameters(modify_parameters_rfm)

    if reform is None:
        sim = Microsimulation(**sim_kwargs)
    else:
        sim = Microsimulation(reform=reform, **sim_kwargs)
    if baseline:
        print("Running current law policy baseline")
    else:
        print("Reform policy is: ", reform)

    sim.year = 2023

    # Check that start_year is appropriate
    if year > DATA_LAST_YEAR:
        raise RuntimeError("Start year is beyond data extrapolation.")

    # define market income - taking expanded_income and excluding gov't
    # transfer benefits
    market_income = sim.calc("household_market_income", period=year).values

    # Compute marginal tax rates (can only do on earned income now)

    # Put MTRs, income, tax liability, and other variables in dict
    length = sim.calc("household_weight", period=year).size
    household = sim.populations["household"]
    person = sim.populations["person"]
    max_age_in_hh = household.max(person("age", str(sim.year)))
    tax_dict = {
        "mtr_labinc": get_household_mtrs(
            reform,
            "employment_income",
            period=year,
            # baseline=sim,
            **sim_kwargs,
        ).values,
        "mtr_capinc": get_household_mtrs(
            reform,
            "interest_income",
            period=year,
            # baseline=sim,
            **sim_kwargs,
        ).values,
        "age": max_age_in_hh,
        "total_labinc": sim.calc(
            "employment_income", map_to="household", period=year
        ).values,
        "total_capinc": sim.calc(
            "interest_income", map_to="household", period=year
        ).values,
        "market_income": market_income,
        "total_tax_liab": sim.calc("household_tax", period=year).values,
        "payroll_tax_liab": sim.calc(
            "employee_payroll_tax", map_to="household", period=year
        ).values,
        "etr": (
            1
            - (
                sim.calc(
                    "household_net_income", map_to="household", period=year
                ).values
            )
            / market_income
        ).clip(-1, 1.5),
        "year": year * np.ones(length),
        "weight": sim.calc("household_weight", period=year).values,
    }

    return tax_dict


def get_data(
    baseline=False,
    start_year=2023,
    reform=None,
    data="cps",
    path=CUR_PATH,
    client=None,
    num_workers=1,
):
    """
    This function creates dataframes of micro data with marginal tax rates and
    information to compute effective tax rates from the PopulationSim object.
    The resulting dictionary of dataframes is returned and saved to disc in a
    pickle file.

    Args:
        baseline (boolean): True if baseline tax policy
        start_year (int): first year of budget window
        reform (PolicyEngine Reform object): IIT policy reform parameters, None
            if baseline
        data (DataFrame or str): DataFrame or path to datafile for the
            PopulationSim object
        path (str): path to save microdata files to
        client (Dask Client object): client for Dask multiprocessing
        num_workers (int): number of workers to use for Dask multiprocessing

    Returns:
        micro_data_dict (dict): dict of Pandas Dataframe, one for each year
            from start_year to the maximum year FiscalSim-US can analyze
        FiscalSimUS_version (str): version of FiscalSim-US used

    """
    # Compute MTRs and taxes or each year, but not beyond DATA_LAST_YEAR
    lazy_values = []
    for year in range(start_year, DATA_LAST_YEAR + 1):
        lazy_values.append(
            delayed(get_calculator_output)(baseline, year, reform, data)
        )
    if client:  # pragma: no cover
        futures = client.compute(lazy_values, num_workers=num_workers)
        results = client.gather(futures)
    else:
        results = compute(
            *lazy_values,
            scheduler=dask.multiprocessing.get,
            num_workers=num_workers,
        )

    # dictionary of data frames to return
    micro_data_dict = {}
    for i, result in enumerate(results):
        year = start_year + i
        df = pd.DataFrame.from_dict(result)
        micro_data_dict[str(year)] = df

    if baseline:
        pkl_path = os.path.join(path, "micro_data_baseline.pkl")
    else:
        pkl_path = os.path.join(path, "micro_data_policy.pkl")

    with open(pkl_path, "wb") as f:
        pickle.dump(micro_data_dict, f)

    # Do some garbage collection
    del results

    # Pull FiscalSim-US version for reference
    FiscalSimUS_version = (
        None  # pkg_resources.get_distribution("taxcalc").version
    )

    return micro_data_dict, FiscalSimUS_version
