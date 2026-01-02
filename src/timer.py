import time


class Timer:
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = 0

    def __enter__(self):
        self.start_time = time.time()
        print(f"[TIMER] {self.name} started...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        print(f"[TIMER] {self.name} completed in {self.elapsed:.2f}s")
        return False

    @staticmethod
    def log(name, elapsed_time):
        print(f"[TIMER] {name} took {elapsed_time:.2f}s")

