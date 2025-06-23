from aux_functions import (
    detect_gait_phase_row, parse_can_log, get_grf_sum_df,
    plot_encoder, plot_imu, plot_grf, plot_grf_angle
)
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


# ===============================
# File paths for Left Leg Trials
# ===============================
left_paths = {
    # "ankle": "recorded_data/left ankel angle 0-45 v2.trc",
    # "hip":   "recorded_data/left hip angle 0-90 v2.trc",
    # "knee":  "recorded_data/left knee angle 0-120 v2.trc",
    # "grf":   "recorded_data/left GRF 4 sensor v2.trc",
    
    "ankle": "recorded_data/left movement v1_filtered.trc",
    # "ankle": "recorded_data/left movement v3.trc",
    # "seat":   "recorded_data/left movement seat only test ankle v5.trc",
    # "seat stand": "recorded_data/left movement seat stand up v4_filtered.trc",
    # "still": "recorded_data/left still test v1.trc"
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

grf_df = get_grf_sum_df(grf_data)
# Convert to list of dicts for grf_list
grf_list = grf_df.to_dict(orient="records")

# ===============================
# Build DataFrames & Sort by Time
# ===============================
knee_df  = pd.DataFrame(knee_list).rename(columns={"Angle": "knee"}).sort_values("TimeOffset")
hip_df   = pd.DataFrame(hip_list).rename(columns={"Angle": "hip"}).sort_values("TimeOffset")
# ankle_df = pd.DataFrame(ankle_list).rename(columns={"Angle": "ankle"}).sort_values("TimeOffset")
imu_left = imu_data.get("0026", [])
imu_df = pd.DataFrame(imu_left).rename(columns={"Pitch (Y)": "femur_pitch", "Roll (X)": "femur_roll", "Yaw (Z)": "femur_yaw"}).sort_values("TimeOffset")

# Merge all column（exclude ankel_angle and grf）
df = pd.merge_asof(knee_df, hip_df, on="TimeOffset", direction="nearest", tolerance=50)
df = pd.merge_asof(df, grf_df, on="TimeOffset", direction="nearest", tolerance=50)
df = pd.merge_asof(df, imu_df, on="TimeOffset", direction="nearest", tolerance=50)

# ===============================
# Add Gait Phase Classification
# ===============================
df["gait_phase"] = df.apply(detect_gait_phase_row, axis=1)

# ===============================
# Visualization (Optional)
# ===============================
# plot_encoder(encoder_data, "Left_Knee_Angle", "Left_Hip_Angle")
# plot_encoder(encoder_data, "Left_Hip_Angle")
# plot_encoder(encoder_data, "Left_Ankle_Angle")

# plot_imu(imu_data, "Left_Femur_RPY_(IMU)")
# plot_grf(grf_data, "Left_Ankle_GRF_Channel_3", "Left_Ankle_GRF_Channel_2", "Left_Ankle_GRF_Channel_4", "Left_Ankle_GRF_Channel_1")
# plot_grf(grf_data, "Left_Ankle_GRF_Channel_2")
# plot_grf(grf_data, "Left_Ankle_GRF_Channel_3")
# plot_grf(grf_data, "Left_Ankle_GRF_Channel_4")

# ===============================
# Plot GRF sum, knee angle, and hip angle together
# ===============================
# plot_grf_angle(df)

# ===============================
# Debug or Export
# ===============================
# 自動選擇存在的欄位進行顯示
cols_to_show = [col for col in ["TimeOffset", "knee", "hip", "ankle", "grf", "femur_pitch", "femur_roll", "femur_yaw", "gait_phase"] if col in df.columns]
print(df[cols_to_show].head())
print(df["gait_phase"].unique())

# 顯示 knee 在 110 到 135 之間的所有資料列
knee_mask = (df["knee"] >= 110) & (df["knee"] < 135)
print(df[knee_mask])

phase_map = {phase: i for i, phase in enumerate(df["gait_phase"].unique())}
df["phase_num"] = df["gait_phase"].map(phase_map)

plt.figure(figsize=(12, 3))
plt.plot(df["TimeOffset"], df["phase_num"], drawstyle="steps-post")
plt.yticks(list(phase_map.values()), list(phase_map.keys()))
plt.xlabel("TimeOffset (ms)")
plt.title("Gait Phase Sequence")
plt.tight_layout()
plt.show()