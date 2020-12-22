from typing import Dict


class EarlyStopper:

    def __init__(self, patience: int, criterion: str, delta: float):
        self.__patience = patience
        self.__remain = patience
        self.__criterion = criterion
        self.__delta = delta

    def check_early_stopping(self, metrics: Dict[str, list]) -> bool:
        history = metrics[self.__criterion]
        if len(history) < 2:
            return
        if history[-2] - history[-1] < self.__delta:
            self.__remain -= 1
        else:
            self.__remain = self.__patience
        return self.__remain < 0

