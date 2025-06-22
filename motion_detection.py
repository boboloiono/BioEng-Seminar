from aux_functions import (
    detect_gait_phase_row, parse_can_log,
    plot_encoder, plot_imu, plot_grf
)
import pandas as pd
from collections import defaultdict

# ===============================
# File paths for Left Leg Trials
# ===============================
left_paths = {
    #"ankle": "recorded_data/left ankel angle 0-45 v2.trc",
    "ankle": "recorded_data/left movement v1.trc",
    # "hip":   "recorded_data/left hip angle 0-90 v2.trc",
    # "knee":  "recorded_data/left knee angle 0-120 v2.trc",
    # "grf":   "recorded_data/left GRF 4 sensor v2.trc"
}

right_paths = {
    "ankle": "recorded_data/right still test v1.trc",
}

# ===============================
# Initialize CAN Data Storage
# ===============================
encoder_data = defaultdict(list)
imu_data     = defaultdict(list)
grf_data     = defaultdict(lambda: defaultdict(list))

# ===============================
# Parse CAN Logs from All Files
# ===============================
for path in left_paths.values():
    parse_can_log(path, encoder_data, imu_data, grf_data)

# ===============================
# Extract Left Leg Sensor Lists
# ===============================
knee_list  = encoder_data["0020"]
hip_list   = encoder_data["0022"]
ankle_list = encoder_data["0024"]
grf_list   = grf_data["0025"]["Channel 1"]  # You may change channel as needed

# ===============================
# Build DataFrames & Sort by Time
# ===============================
knee_df  = pd.DataFrame(knee_list).rename(columns={"Angle": "knee"}).sort_values("TimeOffset")
hip_df   = pd.DataFrame(hip_list).rename(columns={"Angle": "hip"}).sort_values("TimeOffset")
ankle_df = pd.DataFrame(ankle_list).rename(columns={"Angle": "ankle"}).sort_values("TimeOffset")
grf_df   = pd.DataFrame(grf_list).rename(columns={"Value": "grf"}).sort_values("TimeOffset")   # TODO: change to real GRF

# ===============================
# Align TimeSeries via Merge-As-Of
# ===============================
df = pd.merge_asof(knee_df, hip_df, on="TimeOffset", direction="nearest", tolerance=50)
df = pd.merge_asof(df, ankle_df, on="TimeOffset", direction="nearest", tolerance=50)
df = pd.merge_asof(df, grf_df, on="TimeOffset", direction="nearest", tolerance=50)

# ===============================
# Add Gait Phase Classification
# ===============================
df["gait_phase"] = df.apply(detect_gait_phase_row, axis=1)

# ===============================
# Visualization (Optional)
# ===============================
# plot_encoder(encoder_data, "Left_Knee_Angle")
# plot_encoder(encoder_data, "Left_Hip_Angle")
# plot_encoder(encoder_data, "Left_Ankle_Angle")

plot_imu(imu_data, "Left_Femur_RPY_(IMU)")
plot_grf(grf_data, "Left_Ankle_GRF_Channel_1")
plot_grf(grf_data, "Left_Ankle_GRF_Channel_2")
plot_grf(grf_data, "Left_Ankle_GRF_Channel_3")
plot_grf(grf_data, "Left_Ankle_GRF_Channel_4")

# ===============================
# Debug or Export
# ===============================
print(df[["TimeOffset", "knee", "hip", "ankle", "grf", "gait_phase"]].head())
