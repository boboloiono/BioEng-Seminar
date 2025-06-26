from aux_functions import (
    detect_gait_phase, parse_can_log, get_grf_sum_df,
    plot_encoder, plot_imu, plot_grf, plot_grf_angle, filter_df_by_time,
    detect_sin_segment, plot_sin_segment,
    plot_angle_sin_segment, detect_and_fill_nan_segments, detect_and_fill_fixed_segments
)
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


# ===============================
# File paths for Left Leg Trials
# ===============================
paths = {
    # "ankle": "recorded_data/left ankel angle 0-45 v2.trc",
    # "hip":   "recorded_data/left hip angle 0-90 v2.trc",
    # "knee":  "recorded_data/left knee angle 0-120 v2.trc",
    # "grf":   "recorded_data/left GRF 4 sensor v2.trc",
    
    "new_hip": "recorded_data/left movement v9_filtered.trc",
    "old_hip": "recorded_data/left movement v9_filtered.trc",
    "old_ankle": "recorded_data/left movement v9_filtered.trc",
    # "seat":   "recorded_data/left movement seat only test ankle v5.trc",
    # "seat stand": "recorded_data/left movement seat stand up v5.trc",
    # "still": "recorded_data/left still test v1.trc"
}

# ===============================
# Initialize CAN Data Storage
# ===============================
encoder_data_new = defaultdict(list)
encoder_data_old = defaultdict(list)
encoder_data_old_ankle = defaultdict(list)
imu_data     = defaultdict(list)
grf_data     = defaultdict(lambda: defaultdict(list))

# ===============================
# Parse CAN Logs from All Files
# ===============================
parse_can_log(paths["new_hip"], encoder_data_new, imu_data, grf_data)
parse_can_log(paths["old_hip"], encoder_data_old, imu_data, grf_data)
parse_can_log(paths["old_ankle"], encoder_data_old_ankle, imu_data, grf_data)

grf_df = get_grf_sum_df(grf_data)
grf_list = grf_df.to_dict(orient="records")

# ===============================
# Build DataFrames & Sort by Time
# ===============================
knee_df  = pd.DataFrame(encoder_data_new["0020"]).rename(columns={"Angle": "knee"}).sort_values("TimeOffset")
hip_df   = pd.DataFrame(encoder_data_new["0022"]).rename(columns={"Angle": "hip"}).sort_values("TimeOffset")
ankle_df = pd.DataFrame(encoder_data_new["0024"]).rename(columns={"Angle": "ankle"}).sort_values("TimeOffset")
imu_left = imu_data.get("0026", [])
imu_df = pd.DataFrame(imu_left).rename(columns={"Pitch (Y)": "femur_pitch", "Roll (X)": "femur_roll", "Yaw (Z)": "femur_yaw"}).sort_values("TimeOffset")

# 先建立統一的時間軸
all_time = pd.Series(sorted(set(knee_df['TimeOffset']).union(hip_df['TimeOffset']).union(ankle_df['TimeOffset']).union(grf_df['TimeOffset'])))
new_df = pd.DataFrame({'TimeOffset': all_time}) # 以 all_time 為主，merge 各關節
new_df = pd.merge_asof(new_df, knee_df, on='TimeOffset', direction='nearest', tolerance=50)
new_df = pd.merge_asof(new_df, hip_df, on='TimeOffset', direction='nearest', tolerance=50)
new_df = pd.merge_asof(new_df, ankle_df, on='TimeOffset', direction='nearest', tolerance=50)
new_df = pd.merge_asof(new_df, grf_df, on='TimeOffset', direction='nearest', tolerance=50)
new_df = filter_df_by_time(new_df, 15000, 27000)
# 畫出還沒做 signal processing 的 Knee, Hip, Ankle 圖
# plot_encoder(new_df, "knee", "hip", "ankle")

new_df['hip'] = savgol_filter(new_df['hip'], window_length=15, polyorder=3, mode='interp')  # 保留 hip 平滑
new_df['ankle'] = savgol_filter(new_df['ankle'], window_length=15, polyorder=3, mode='interp')  # 註解掉 ankle 平滑

# 舊檔案 hip
old_df = pd.DataFrame(encoder_data_old["0022"]).interpolate(method='cubic').rename(columns={"Angle": "hip"}).sort_values("TimeOffset")
old_df = filter_df_by_time(old_df, 16500, 18800)
old_df['hip'] = savgol_filter(old_df['hip'], window_length=61, polyorder=3, mode='interp')
# 先偵測 sin 波 pattern
sin_start, sin_end = detect_sin_segment(old_df, hip_col='hip', time_col='TimeOffset', window=2000)
sin_pattern = old_df.iloc[sin_start:sin_end].copy()
# 畫出 old_df 的 sin 波擷取處
# plot_sin_segment(old_df, sin_start, sin_end, 'hip', 'old_df hip 及 sin波擷取區段')

# 讀取 old_ankle 資料
old_ankle_df = pd.DataFrame(encoder_data_old_ankle["0024"]).interpolate(method='cubic').rename(columns={"Angle": "ankle"}).sort_values("TimeOffset")
old_ankle_df = filter_df_by_time(old_ankle_df, 16000, 18200)
old_ankle_df['ankle'] = savgol_filter(old_ankle_df['ankle'], window_length=61, polyorder=3, mode='interp')
ankle_sin_start, ankle_sin_end = detect_sin_segment(old_ankle_df, hip_col='ankle', time_col='TimeOffset', window=1800)
ankle_sin_pattern = old_ankle_df.iloc[ankle_sin_start:ankle_sin_end].copy()
# 畫出 old_ankle_df 的 sin 波擷取處
# plot_sin_segment(old_ankle_df, ankle_sin_start, ankle_sin_end, 'ankle', 'old_ankle_df ankle 及 sin波擷取區段')

# 找出 new_df['ankle'] 連續出現 fixed_value_list 中任一值的區段，僅對超過 2 點的區段補 sin 波，其餘用 cubic interpolation
ankle_values = new_df['ankle'].values  # 確保 ankle_values 已定義
fixed_value_list = []
if len(ankle_values) > 0:
    prev_val = ankle_values[0]
    start = 0
    for i in range(1, len(ankle_values)):
        if not np.isclose(ankle_values[i], prev_val):
            if i - start > 100:
                fixed_value_list.append(prev_val)
            start = i
            prev_val = ankle_values[i]
    # 檢查最後一段
    if len(ankle_values) - start > 2:
        fixed_value_list.append(prev_val)

# 找出 new_df['hip'] 為 NaN 的連續區段，僅對超過 500ms 的區段補 sin 波，其餘用 cubic interpolation
new_df, hip_segments = detect_and_fill_nan_segments(new_df, 'hip', sin_pattern['hip'].values, min_duration=500)
# 找出 new_df['ankle'] 為定值的連續區段，僅對超過 500ms 的區段補 sin 波，其餘用 cubic interpolation
new_df, ankle_segments = detect_and_fill_fixed_segments(new_df, 'ankle', fixed_value_list, ankle_sin_pattern['ankle'].values, min_duration=500)

ankle_fixed_mask = np.isin(new_df['ankle'], fixed_value_list)
# 註解掉所有 Angle 欄位（knee、ankle）的補值與平滑處理，保留 hip
new_df['knee'] = new_df['knee'].interpolate(method='cubic')
new_df['hip'] = new_df['hip'].interpolate(method='cubic')
new_df['ankle'] = new_df['ankle'].interpolate(method='cubic')
new_df['knee'] = savgol_filter(new_df['knee'], window_length=151, polyorder=3, mode='interp')
for _ in range(30):
    new_df['hip'] = savgol_filter(new_df['hip'], window_length=151, polyorder=3, mode='interp')
    new_df['ankle'] = savgol_filter(new_df['ankle'], window_length=151, polyorder=3, mode='interp')

# 畫圖顯示 new_df new angle (knee 用 sin 波補值的區段（紅色），ankle 用藍色)
# plot_angle_sin_segment(new_df, hip_segments, ankle_segments)

# ===============================
# Add Gait Phase Classification
# ===============================
new_df['gait_phase'] = new_df.apply(lambda row: detect_gait_phase(
    grf=row['grf'],
    knee=row['knee'],
    hip=row['hip'],
    ankle=row['ankle']), axis=1)

# ===============================
# Visualization
# ===============================
# plot_imu(imu_data, "Left_Femur_RPY_(IMU)")
# plot_grf(new_df)
plot_grf_angle(new_df, "grf", "knee", "hip", "ankle", "gait_phase")

# ===============================
# Debug or Export
# ===============================
cols_to_show = [col for col in ["TimeOffset", "knee", "hip", "ankle", "grf", "gait_phase"] if col in new_df.columns]
print(new_df[cols_to_show].head())
print(new_df["gait_phase"].unique())