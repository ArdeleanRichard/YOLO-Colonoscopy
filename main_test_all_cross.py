import torch

from constants import DATA, results_root_test, ALL_MODELS, DATA_TEST
from functions import ModelEvaluator

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    RESULTS_CSV_PATH = f"{results_root_test}/results.csv"

    for MODEL_NAME in ALL_MODELS:
        evaluator = ModelEvaluator(MODEL_NAME, DATA_TEST, device, conf=0.25, iou=0.5, mode="cross")
        evaluator.load_model()

        # Evaluate the model
        evaluator.evaluate()

        # Print metrics
        evaluator.print_metrics()

        # Save results to CSV
        evaluator.save_results_to_csv(RESULTS_CSV_PATH)