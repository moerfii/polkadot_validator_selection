import numpy as np
import pandas as pd

"""
The game is as such:
There are validators and nominators.
The nominators all have the same stake available.
The nominators place their stake in the nominators.
The validators set a commission, which ranges from 0% to 100%.
The validators get rewards of 100$ if they get elected.
In order for them to be elected they have to have a big enough minimum stake backing them.
The nominators share the reward left over after commission based on their relative share of the total stake.
Nominators may select up to 16 validators they wish to back.
It should return the optimal set of validators for one validator.
"""


def accumulate_stakes(nominator_dataframe, validator_dataframe):
    for index, row in nominator_dataframe.iterrows():
        for validator_index in row["validator_list"]:
            validator_dataframe.at[validator_index, "stake"] += row["stake"]
            validator_dataframe.at[validator_index, "nominators"] += 1
    return validator_dataframe


def setup_game():
    # create a dataframe with 1000 validators
    validators = pd.DataFrame({"validator_id": range(1000)})

    # generate normal distribution of commission rates with a mean at 10%
    validators["commission"] = np.random.normal(loc=10, scale=10, size=1000)

    # ensure that commission rates are between 0% and 100%
    validators["commission"] = validators["commission"].clip(lower=0, upper=100)

    # round commission rates to 1 decimal point and take absolute value
    validators["commission"] = validators["commission"].abs().round(1) / 100

    # sort values by commission in order to attribute more nominators to them
    validators.sort_values("commission", ascending=True, inplace=True)

    # init stakes of 0 for later
    validators["stake"] = 0

    # init nominator_counter for later
    validators["nominators"] = 0

    # create a dataframe with 2000 nominators
    nominators = pd.DataFrame({"nominator_id": range(1000, 3000)})

    # generate random stake for each nominator between 100 and 10000 DOT
    nominators["stake"] = np.random.uniform(low=100, high=10000, size=2000)
    nominators["stake"] = nominators["stake"].abs().round(0)

    nominators["validator_list"] = [np.empty(0, dtype=float)] * len(nominators)

    for index, row in nominators.iterrows():
        number_of_validators = np.random.poisson(
            lam=15
        )  # set mean to 15 since most nominators should opt for 16 validators
        number_of_validators = np.clip(number_of_validators, 1, 16)
        counter_selected_validators = 0
        array_selected_validators = np.full(number_of_validators, -1)
        while counter_selected_validators < number_of_validators:
            validator_row_index = int(abs(np.random.normal(loc=100, scale=200, size=1)))
            while validator_row_index in array_selected_validators:
                validator_row_index = int(
                    abs(np.random.normal(loc=100, scale=200, size=1))
                )
            array_selected_validators[counter_selected_validators] = validators.iloc[
                [validator_row_index]
            ]["validator_id"]
            counter_selected_validators += 1
        nominators.at[index, "validator_list"] = array_selected_validators

    # sort validators again by index for quicker updating
    validators.sort_values("validator_id", ascending=True, inplace=True)
    validators = accumulate_stakes(nominators, validators)
    return nominators, validators


def solve_game(nominator_dataframe, validator_dataframe, nominator_stake):
    """
    :param Dataframe nominator_dataframe: Matrix of nominator id, bonded stake and their validator preference list
    :param Dataframe validator_dataframe: Matrix of validator id, commission and the total stake backing them.
    :param int nominator_stake: The simulated stake bonded by us
    :return: list suggested_validator_list

    In a first step this solver should aim to reduce the list of potential validators by getting rid of validators who
    did not reach the minimum stake of the previous round. This step is unclear since the sequential phragmen algorithm
    might reach a local optimum different from the one calculated by us. The issue here is that the stake distribution
    might look totally different. For this simplified game, we've reduced the validator list by sorting them by total
    stake backing them and reducing the list to the top 150 backed.
    In a second step it solves for the most profitable of the remaining set.
    What does most profitable mean?
        - choose validators where the relative share of own stake is maximal.
        - choose validators where commission is minimal
    """

    # default reward, is more or less the same for every validator
    reward = 100

    # sort validators and cut off at threshold
    threshold = 150
    validator_dataframe.sort_values("stake", ascending=False, inplace=True)
    validator_dataframe_reduced = validator_dataframe.head(threshold)

    # init column for stake share and expected return
    validator_dataframe_reduced["stake_share"] = 0
    validator_dataframe_reduced["expected_return"] = 0

    validator_dataframe_reduced["stake_share"] = (
        nominator_stake / validator_dataframe_reduced["stake"]
    )
    validator_dataframe_reduced["expected_return"] = (
        reward - (validator_dataframe_reduced["commission"] * reward)
    ) * validator_dataframe_reduced["stake_share"]

    validator_dataframe_reduced.sort_values(
        "expected_return", ascending=False, inplace=True
    )
    top_validators = validator_dataframe_reduced.head(16)
    suggested_validator_list = []
    for index, row in top_validators.iterrows():
        suggested_validator_list.append(row["validator_id"])
    return suggested_validator_list  # return a list of validators to select (optimally len 16)


if __name__ == "__main__":
    nominator_dataframe, validator_dataframe = setup_game()
    stake = 10000
    print(solve_game(nominator_dataframe, validator_dataframe, stake))
