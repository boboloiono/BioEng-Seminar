input_path = "recorded_data/left movement v9.trc"
output_path = "recorded_data/left movement v9_filtered.trc"
threshold = 16000  # ms

import re

pattern = re.compile(r"\s*\d+\)\s+([\d.]+)\s+Rx\s+([0-9a-fA-F]{4})")

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        match = pattern.match(line)
        if match:
            time_offset = float(match.group(1))
            if time_offset >= 10669 and time_offset <= 25500:
                outfile.write(line)
        else:
            outfile.write(line)