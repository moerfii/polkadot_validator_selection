import unittest
import pandas as pd
from src.adjustment import AdjustmentTool

class AdjustmentTest(unittest.TestCase):
    def test_even_split_strategy(self):
        test_dataframe = pd.DataFrame(
            {
                "nominator": ["nominator1", "nominator1", "nominator1"],
                "validator": ["validator1", "validator2", "validator3"],
                "proportional_bond": [40, 40, 40],
                "total_bond": [120, 120, 120],
                "number_of_validators": [3, 3, 3],
                "total_proportional_bond": [40, 40, 40],
                "era": [1, 1, 1],
                "solution_bond": [60, 30, 30],
                "prediction": [30, 30, 40],
            }
        )
        adjustment_tool = AdjustmentTool()
        adjusted_dataframe = adjustment_tool.even_split_strategy(test_dataframe)
        self.assertEqual(adjusted_dataframe["prediction"].sum(), 120)


if __name__ == '__main__':
    unittest.main()
