# scripts/utils/utils.py

def calculate_accuracy(correct, total):
    if total == 0:
        return 0
    return (correct / total) * 100
