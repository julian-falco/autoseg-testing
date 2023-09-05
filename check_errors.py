import os
import json

os.chdir(os.path.join("..", "testing_log", "oblique"))

errors = 0
total = 0
times = []
for fname in os.listdir("."):
    with open(fname, "r") as f:
        total += 1
        log_data = json.load(f)
        if log_data["exception"]:
            errors += 1
        else:
            times.append(float(log_data["time_elapsed"]))
            print(fname)

print(f"Average time per case: {sum(times)/(total-errors)}")
print(f"Errors: {errors}/{total}")