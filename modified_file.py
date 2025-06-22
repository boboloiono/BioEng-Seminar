input_path = "recorded_data/left movement seat stand up v4.trc"
output_path = "recorded_data/left movement seat stand up v4_filtered.trc"
threshold = 10000  # ms

import re

pattern = re.compile(r"\s*\d+\)\s+([\d.]+)\s+Rx\s+([0-9a-fA-F]{4})")

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        match = pattern.match(line)
        if match:
            time_offset = float(match.group(1))
            if time_offset >= threshold:
                outfile.write(line)
        else:
            # 若有標頭或其他格式，視情況保留
            outfile.write(line)