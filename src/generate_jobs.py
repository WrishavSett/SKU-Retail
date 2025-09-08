import os
from datetime import datetime, timedelta

# Settings
start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 12, 31)
log_mode = "separate"  # "separate" = one log per month, "combined" = all in one log
log_dir = "logs"
log_file = os.path.join(log_dir, "all.log")

# Ensure logs directory exists
os.makedirs(log_dir, exist_ok=True)

# Generate list of months between start and end
months = []
current = start_date
while current <= end_date:
    months.append(current.strftime("%b-%y").lower())  # e.g. jan-22
    # move to next month
    if current.month == 12:
        current = datetime(current.year + 1, 1, 1)
    else:
        current = datetime(current.year, current.month + 1, 1)

# Build nohup commands
commands = []
for m in months:
    dataset_file = f"dataset/transaction/{m}.csv"
    if log_mode == "separate":
        cmd = (
            f'nohup python3 app/main.py --file "{dataset_file}" '
            f'--use-llm --llm-model "qwen2.5:7b" > {log_dir}/{m}.log 2>&1 &'
        )
    else:  # combined log
        cmd = (
            f'echo "===== Running {m} =====" >> {log_file} && '
            f'nohup python3 app/main.py --file "{dataset_file}" '
            f'--use-llm --llm-model "qwen2.5:7b" >> {log_file} 2>&1 &'
        )
    commands.append(cmd)

# Write to a shell script
with open("run_all.sh", "w") as f:
    f.write("#!/bin/bash\n\n")
    for cmd in commands:
        f.write(cmd + "\n")

print("Generated run_all.sh with all nohup commands.")
print(f"Log mode: {log_mode} -> logs in {log_dir}/")
