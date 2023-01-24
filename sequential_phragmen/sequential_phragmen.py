from typing import List, Dict


def sequential_phragmen(validators: List[int], nominators: Dict[int, int], votes: Dict[int, List[int]],
                        output_size: int, max_iter: int = 100) -> List[int]:
    # Initialize the stakes and variances for each validator
    stakes = [[0] * len(nominators) for _ in range(len(validators))]
    variances = [0] * len(validators)

    # Iterate until convergence or until the maximum number of iterations is reached
    for i in range(max_iter):
        # Calculate the total stake and the stake for each validator
        total_stake = sum(nominators.values())
        for j, val in enumerate(validators):
            for k, nom in enumerate(nominators):
                if val in votes[nom]:
                    stakes[j][k] = nominators[nom]
                else:
                    stakes[j][k] = 0

        # Calculate the variance of the stake for each validator
        for j, val in enumerate(validators):
            variance = 0
            for k, nom in enumerate(nominators):
                variance += (stakes[j][k]/total_stake - 1/len(validators))**2
            variances[j] = variance

        # Find the validator with the minimum stake
        min_index = min(range(len(validators)), key=lambda i: sum(stakes[i]))

        # Find the nominator with the maximum stake behind the validator with the minimum stake
        max_stake_index = max(range(len(nominators)), key=lambda i: stakes[min_index][i])
        max_stake = nominators[max_stake_index]

        # Check if the maximum stake is greater than the minimum stake
        if max_stake > sum(stakes[min_index]):
            # Transfer the excess stake from the nominator with the maximum stake to the validator with the minimum stake
            nominators[max_stake_index] -= (max_stake - sum(stakes[min_index]))
            stakes[min_index][max_stake_index] = max_stake
        else:
            # If the maximum stake is not greater than the minimum stake, we have reached convergence
            break

    # Return the list of validators with the optimal solution
    return [validators[i] for i in range(output_size) if sum(stakes[i]) > 0]



validators = [1,2,3,4]
# Dictionary of nominators and their total bond
nominators = {
    1: 50,
    2: 100,
    3: 200,
    4: 300,
    5: 400
}

# Dictionary of validator indexes that each nominator voted for
votes = {
    1: [3, 4],
    2: [1, 2],
    3: [2, 3],
    4: [1, 3],
    5: [4, 1]
}

# Call the sequential_phragmen() function
output_size = 2
optimal_validators = sequential_phragmen(validators, nominators, votes, output_size)

# Print the list of optimal validators
print(optimal_validators)
