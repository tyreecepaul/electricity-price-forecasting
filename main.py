from data_cleaning.data import Data, getTrainingData, getTestData
import datetime

def main() -> None:
    # Get train test split
    train: Data = getTrainingData(
        regions=["NSW1"], 
        start=datetime(2024, 1, 1), 
        end=datetime(2024, 12, 31)
    )
    test: Data = getTestData(
        regions=["NSW1"], 
        start=datetime(2024, 1, 1), 
        end=datetime(2024, 12, 31)
    )

    # Pass training data into models and train

    # Get predictions using training data and test data

    # Output results
    pass

main()
