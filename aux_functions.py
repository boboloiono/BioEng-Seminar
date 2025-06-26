###############################################################
##  Filename: aux_functions.py
##  Author: Pei-Yu Lin
##  Description:
##    This file includes:
##     1. CAN ID Labels and ID Category
##     2. Raw data decoding functions
##     3. Gait phase detection logic
##     4. Parse CAN data into encoder, IMU and GRF depending on CAN ID
##     5. Plotting functions for encoder, IMU, and GRF data
###############################################################

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks

# =============================================================
# CAN ID Labels and ID group sets
# =============================================================

encoder_id_label = {
    "Right_Knee_Angle": "0010",
    "Right_Hip_Angle": "0012",
    "Right_Ankle_Angle": "0014",    # no signal
    "Left_Knee_Angle": "0020",
    "Left_Hip_Angle": "0022",
    "Left_Ankle_Angle": "0024",     # no signal
}

imu_id_label = {
    "Right_Femur_RPY_(IMU)": "0016",
    "Left_Femur_RPY_(IMU)": "0026"
}

grf_id_label = {
    # Left foot
    "Left_Ankle_GRF_Channel_1": ("0025", "Channel 1"),
    "Left_Ankle_GRF_Channel_2": ("0025", "Channel 2"),
    "Left_Ankle_GRF_Channel_3": ("0025", "Channel 3"),
    "Left_Ankle_GRF_Channel_4": ("0025", "Channel 4"),
    # Right foot
    "Right_Ankle_GRF_Channel_1": ("0015", "Channel 1"),
    "Right_Ankle_GRF_Channel_2": ("0015", "Channel 2"),
    "Right_Ankle_GRF_Channel_3": ("0015", "Channel 3"),
    "Right_Ankle_GRF_Channel_4": ("0015", "Channel 4")
}

# CAN ID group sets
encoder_ids = {"0010", "0012", "0014", "0020", "0022", "0024"}
knee_ids = {"0010", "0020"}
hip_ids = {"0012", "0022"}
ankle_ids = {"0014", "0024"}
imu_ids = {"0016", "0026"}
grf_ids = {"0015", "0025"}
stiffness_ids = {"0011", "0013", "0021", "0023"}

# =============================================================
# 2. Raw Data Decoding Functions
# =============================================================

# Angle and Stiffness
def decode_phi_hip(data_high, data_low):
    # 14-bit raw data, angle range [0°,180°] or [-90°,90°]
    K = 180 / (2**14 - 1) * 4
    if data_high != 0 and data_low != 0: 
        raw_value = (data_high << 8) | data_low
        phi = raw_value * K - 320
        return phi
    else:
        return None

def decode_phi_knee(data_high, data_low):
    # 14-bit raw data, angle range [0°,180°] or [-90°,90°]
    K = 180 / (2**14 - 1) * 4
    if data_high != 0 and data_low != 0: 
        raw_value = (data_high << 8) | data_low
        phi = raw_value * K  - 300
        return phi
    else:
        return None

def decode_phi_ankle(data_high, data_low):
    # 14-bit raw data, angle range [0°,180°] or [-90°,90°]
    K = 180 / (2**14 - 1) * 4
    if data_high != 0 and data_low != 0: 
        raw_value = (data_high << 8) | data_low
        phi = raw_value * K  - 300
        return phi
    else:
        return None

def decode_imu_left(data):
    z_val = data[0] | (data[1] << 8)
    x_val = data[2] | (data[3] << 8)
    y_val = data[4] | (data[5] << 8)
    scale = 360.0 / 65535
    return {
        'yaw': z_val * scale,
        'roll': x_val * scale,
        'pitch': -y_val * scale
    }

def decode_imu_right(data):
    z_val = data[0] | (data[1] << 8)
    x_val = data[2] | (data[3] << 8)
    y_val = data[4] | (data[5] << 8)
    scale = 360.0 / 65535
    return {
        'yaw': -z_val * scale,          # [0°, 360°] => [-360°, 0°]
        'roll': -x_val * scale + 180,   # [0°, 360°] => [-360°, 0°] by + 180 => [-180°, +180°]
        'pitch': -y_val * scale + 180   # [0°, 360°] => [-360°, 0°] by + 180 => [-180°, +180°]
    }

def decode_GRF(byte_high, byte_low, v_ref=5.0, series_resistor=1000):
    # Step 1: Combine bytes to get raw ADC value
    raw = (byte_high << 8) | byte_low  # 10-bit expected: max 1024
    
    # Step 2: Convert 10-bit ADC raw value to voltage
    voltage = raw * 5.0 / 1024.0

    # Step 3: Convert voltage to resistance using a voltage divider model
    if voltage > 5.0:
        return 0    # Protection condition: invalid input
    else:
        if voltage == 0:
            resistance_ohm = np.inf  # Avoid division by zero; assume open circuit
        resistance_ohm = (series_resistor * (v_ref - voltage)) / voltage
    
        # Step 4: Map resistance to weight (gram) using a linear approximation
        # if resistance_ohm < 250:
        #     weight_g = 0
        # else:
        weight_g = resistance_ohm * 24

        # Step 5: Convert weight in grams to force in newtons (N)
        force_n = (weight_g / 1000.0) * 9.81  # g → kg → N
        
        return force_n

# =============================================================
# 3. Gait Phase Detection
# =============================================================

def detect_gait_phase(grf, knee, hip, ankle):
    """
    Parameters:
        grf (float): GRF sum (N)
        knee (float): Knee angle (deg)
        hip (float): Hip angle (deg)
        ankle (float): Ankle angle (deg)

    Returns:
        str: Gait phase name
    """
    # 判斷 stance vs swing
    if grf > 450:
        if  ankle > -58:
            return 'Terminal Stance'
        else: #if -22 < knee and -55 < ankle < -45:
            return 'Mid stance'
    if grf > 30:  # stance phase
        if -27 < knee < -18 and ankle > -45:
            return 'Initial Contact'
        elif knee >= -25 and hip > 10:
            return 'Loading Response'
        else:
            return 'Pre swing'
    else:
        if ankle < -65:
            return 'Initial Swing'
        elif -65 <= ankle < -55:
            return 'Mid Swing'
        else:
            return 'Terminal Swing'


# =============================================================
# 4. Parse CAN data
# =============================================================

def parse_can_log(file_path, encoder_data, imu_data, grf_data):
    """
    Parse a .trc CAN log file and fill in encoder_data, imu_data, grf_data.

    Parameters:
        file_path:    str, path to the .trc CAN log file
        encoder_data: defaultdict(list), output for encoder angles
        imu_data:     defaultdict(list), output for IMU orientation
        grf_data:     defaultdict(dict(list)), output for GRF channels
    """
        
    pattern = re.compile(
        r"\s*\d+\)\s+([\d.]+)\s+Rx\s+([0-9a-fA-F]{4})\s+\d+\s+((?:[0-9a-fA-F]{2}\s+){1,8})"
    )


    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            match = pattern.match(line)
            if not match:
                continue

            time_offset = float(match.group(1))
            msg_id = match.group(2).upper().zfill(4)
            data_bytes = match.group(3).strip().split()
            data = [int(b, 16) for b in data_bytes]

            # Encoder (5 bytes)
            if msg_id in knee_ids and len(data) == 5:
                phi = decode_phi_knee(data[3], data[2])
                encoder_data[msg_id].append({
                    "TimeOffset": time_offset,
                    "Angle": phi
                })
            elif msg_id in hip_ids and len(data) == 5:
                phi = decode_phi_hip(data[3], data[2])
                encoder_data[msg_id].append({
                    "TimeOffset": time_offset,
                    "Angle": phi
                })
            elif msg_id in ankle_ids and len(data) == 5:
                phi = decode_phi_ankle(data[3], data[2])
                encoder_data[msg_id].append({
                    "TimeOffset": time_offset,
                    "Angle": phi
                })

            # IMU (6 bytes)
            elif msg_id in imu_ids and len(data) == 6:
                if msg_id == "0016":
                    result = decode_imu_right(data)
                elif msg_id == "0026":
                    result = decode_imu_left(data)
                imu_data[msg_id].append({
                    "TimeOffset": time_offset,
                    "Roll (X)": result['roll'],
                    "Pitch (Y)": result['pitch'],
                    "Yaw (Z)": result['yaw']
                })

            # GRF (8 bytes = 4 channels)
            elif msg_id in grf_ids and len(data) == 8:
                for ch in range(4):
                    grf = decode_GRF(data[ch * 2 + 1], data[ch * 2])
                    grf_data[msg_id][f"Channel {ch + 1}"].append({
                        "TimeOffset": time_offset,
                        "Value": grf
                    })


# =============================================================
# 5. Plotting Functions
# =============================================================

def plot_encoder(df, knee_col, hip_col=None, ankle_col=None):
    """
    Plot encoder angles from a merged DataFrame.
    Parameters:
        df: pandas.DataFrame, must包含 TimeOffset 及指定的角度欄位
        knee_col: str, 膝關節角度欄位名稱
        hip_col: str or None, 髖關節角度欄位名稱
        ankle_col: str or None, 踝關節角度欄位名稱
    """
    plt.figure()
    if knee_col in df:
        plt.plot(df["TimeOffset"], df[knee_col], marker='o', label="Knee")
    if hip_col and hip_col in df:
        plt.plot(df["TimeOffset"], df[hip_col], marker='x', label="Hip (-180)")
    if ankle_col and ankle_col in df:
        plt.plot(df["TimeOffset"], df[ankle_col], marker='^', label="Ankle")
    plt.title("Raw data of Knee/Hip/Ankle Angle vs Time")
    plt.xlabel("Time (ms)")
    plt.ylabel("Joint Angle (deg)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_imu(data, name):
    if name not in imu_id_label:
        print(f"[Error] Unknown IMU name: {name}")
        return
    can_id = imu_id_label[name]
    if can_id not in data or not data[can_id]:
        print(f"[Warning] No data for {name} (CAN ID {can_id})")
        return
    df = pd.DataFrame(data[can_id])
    for axis in ["Roll (X)", "Pitch (Y)", "Yaw (Z)"]:
        if axis not in df.columns:
            print(f"[Warning] Missing {axis} in data for {name}")
            continue
        plt.figure()
        plt.plot(df["TimeOffset"], df[axis], label=axis)
        plt.title(f"{name} - {axis} - CAN ID {can_id}")
        plt.xlabel("Time (ms)")
        plt.ylabel("Angle (deg)")
        plt.grid(True)
        plt.tight_layout()
    plt.show()

def plot_grf(df, sum="grf", ch1_col="ch1", ch2_col="ch2", ch3_col="ch3", ch4_col="ch4", sum_cols=None):
    channel_labels = [sum, ch1_col, ch3_col, ch2_col, ch4_col]
    labels_name = ["sum", "heel", "front_outer", "front_inner", "toe"]
    for col, name in zip(channel_labels, labels_name):
        plt.plot(df["TimeOffset"], df[[col]], label=name, linestyle="--")
    plt.title("Insole FSRs measurement")
    plt.xlabel("Time (ms)")
    plt.ylabel("GRF (N)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_grf_sum_df(grf_data):
    """
    將 grf_data 四個 channel 對齊並加總，回傳 DataFrame (TimeOffset, grf)
    """
    ch1 = pd.DataFrame(grf_data["0025"]["Channel 1"])[["TimeOffset", "Value"]].rename(columns={"Value": "ch1"})
    ch2 = pd.DataFrame(grf_data["0025"]["Channel 2"])[["TimeOffset", "Value"]].rename(columns={"Value": "ch2"})
    ch3 = pd.DataFrame(grf_data["0025"]["Channel 3"])[["TimeOffset", "Value"]].rename(columns={"Value": "ch3"})
    ch4 = pd.DataFrame(grf_data["0025"]["Channel 4"])[["TimeOffset", "Value"]].rename(columns={"Value": "ch4"})
    grf_df = pd.merge_asof(ch1.sort_values("TimeOffset"), ch2.sort_values("TimeOffset"), on="TimeOffset", direction="nearest", tolerance=20)
    grf_df = pd.merge_asof(grf_df, ch3.sort_values("TimeOffset"), on="TimeOffset", direction="nearest", tolerance=20)
    grf_df = pd.merge_asof(grf_df, ch4.sort_values("TimeOffset"), on="TimeOffset", direction="nearest", tolerance=20)
    for _ in range(10):
        grf_df["ch1"] = savgol_filter(grf_df["ch1"], window_length=151, polyorder=3, mode='interp')
        grf_df["ch2"] = savgol_filter(grf_df["ch2"], window_length=151, polyorder=3, mode='interp')
        grf_df["ch3"] = savgol_filter(grf_df["ch3"], window_length=151, polyorder=3, mode='interp')
        grf_df["ch4"] = savgol_filter(grf_df["ch4"], window_length=151, polyorder=3, mode='interp')
    grf_df = grf_df.dropna()
    grf_df["grf"] = grf_df[["ch1", "ch2", "ch3", "ch4"]].sum(axis=1)
    return grf_df[["TimeOffset", "grf", "ch1", "ch2", "ch3", "ch4"]]

def plot_grf_angle(df, name1, name2, name3, name4, phase_col=None):
    """
    phase_col: gait phase 欄位名稱（如 'gait_phase'），
    會畫大區間底色（Swing/Stance），Swing 內畫 Initial/Mid/Terminal 小色帶，Stance 內畫五個 phase 小色帶並標註
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 畫角度
    ax1.plot(df["TimeOffset"], df[name2], label="Knee Angle", color="tab:orange", linestyle="--")
    ax1.plot(df["TimeOffset"], df[name3], label="Hip Angle", color="tab:purple", linestyle="--")
    ax1.plot(df["TimeOffset"], df[name4], label="Ankle Angle", color="tab:green", linestyle="--")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Angle (deg)")
    ax1.tick_params(axis='y')

    # 先計算原始 ylim 並將 y 軸上限提高 20%
    ylim = ax1.get_ylim()
    y_bottom = ylim[0]
    y_top = ylim[1]
    y_range = y_top - y_bottom
    new_y_top = y_top + 0.3 * y_range
    ax1.set_ylim(y_bottom, new_y_top)
    y_text = new_y_top - 0.05 * (new_y_top - y_bottom)

    if phase_col and phase_col in df.columns:
        phases = df[phase_col].values
        times = df["TimeOffset"].values
        swing_set = {"Initial Swing", "Mid Swing", "Terminal Swing"}
        stance_set = {"Loading Response", "Mid stance", "Terminal Stance", "Initial Contact", "Pre swing"}
        swing_label_map = {
            "Initial Swing": "Initial",
            "Mid Swing": "Mid",
            "Terminal Swing": "Terminal"
        }
        swing_color_map = {
            "Initial Swing": "#b2ebf2",
            "Mid Swing": "#80deea",
            "Terminal Swing": "#4dd0e1"
        }
        stance_label_map = {
            "Loading Response": "Loading\nResponse",
            "Mid stance": "Mid\nstance",
            "Terminal Stance": "Terminal\nStance",
            "Initial Contact": "Initial\nContact",
            "Pre swing": "Pre\nswing"
        }
        stance_color_map = {
            "Loading Response": "#ffe082",
            "Mid stance": "#ffd54f",
            "Terminal Stance": "#ffca28",
            "Initial Contact": "#fff9c4",
            "Pre swing": "#ffe0b2"
        }
        phase_type = "Swing" if phases[0] in swing_set else "Stance"
        start_idx = 0
        colors = {"Swing": "#b3e5fc", "Stance": "#ffe0b2"}

        # 畫大色帶
        for i in range(1, len(phases)):
            this_type = "Swing" if phases[i] in swing_set else "Stance"
            if this_type != phase_type:
                ax1.axvspan(times[start_idx], times[i-1], color=colors[phase_type], alpha=0.15, zorder=0)
                mid_time = (times[start_idx] + times[i-1]) / 2
                ax1.text(mid_time, y_text, phase_type, ha='center', va='top', fontsize=14, fontweight='bold', clip_on=True, zorder=2)
                start_idx = i
                phase_type = this_type
        # 畫最後一段
        ax1.axvspan(times[start_idx], times[-1], color=colors[phase_type], alpha=0.15, zorder=0)
        mid_time = (times[start_idx] + times[-1]) / 2
        ax1.text(mid_time, y_text, phase_type, ha='center', va='top', fontsize=14, fontweight='bold', clip_on=True, zorder=2)

        # 畫 Swing 內 Initial/Mid/Terminal 小色帶與標註，Stance 內五個 phase 小色帶與標註
        i = 0
        while i < len(phases):
            if phases[i] in swing_set:
                swing_phase = phases[i]
                swing_start = i
                while i + 1 < len(phases) and phases[i+1] == swing_phase:
                    i += 1
                swing_end = i
                ax1.axvspan(times[swing_start], times[swing_end], color=swing_color_map.get(swing_phase, "#b2ebf2"), alpha=0.5, zorder=1)
                swing_label = swing_label_map.get(swing_phase, "")
                if swing_label:
                    mid_time = (times[swing_start] + times[swing_end]) / 2
                    ax1.text(mid_time, y_text - 0.07 * (new_y_top - y_bottom), swing_label, ha='center', va='top', fontsize=10, fontweight='bold', color='tab:blue', clip_on=True, zorder=3)
            elif phases[i] in stance_set:
                stance_phase = phases[i]
                stance_start = i
                while i + 1 < len(phases) and phases[i+1] == stance_phase:
                    i += 1
                stance_end = i
                ax1.axvspan(times[stance_start], times[stance_end], color=stance_color_map.get(stance_phase, "#ffe082"), alpha=0.5, zorder=1)
                stance_label = stance_label_map.get(stance_phase, "")
                if stance_label:
                    mid_time = (times[stance_start] + times[stance_end]) / 2
                    ax1.text(mid_time, y_text - 0.07 * (new_y_top - y_bottom), stance_label, ha='center', va='top', fontsize=10, fontweight='bold', color='saddlebrown', clip_on=True, zorder=3)
            i += 1

    # 右軸 GRF
    ax2 = ax1.twinx()
    ax2.plot(df["TimeOffset"], df[name1], label="GRF Sum", color="tab:blue", linewidth=3)
    ax2.set_ylabel("GRF (N)")
    ax2.tick_params(axis='y')

    # 提升 GRF y 軸上限 30%
    grf_ylim = ax2.get_ylim()
    grf_bottom = grf_ylim[0]
    grf_top = grf_ylim[1]
    grf_range = grf_top - grf_bottom
    new_grf_top = grf_top + 0.3 * grf_range
    ax2.set_ylim(grf_bottom, new_grf_top)

    # 合併圖例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")

    plt.title("Stride (Gait Cycle)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def filter_df_by_time(df, start_time, end_time=None):
    """
    回傳 start_time <= TimeOffset <= end_time 的資料，若 end_time 為 None 則只用 start_time 篩選
    """
    if end_time is not None:
        return df[(df["TimeOffset"] >= start_time) & (df["TimeOffset"] <= end_time)].reset_index(drop=True)
    else:
        return df[df["TimeOffset"] >= start_time].reset_index(drop=True)

def detect_sin_segment(df, hip_col='hip', time_col='TimeOffset', min_period=100, max_period=2000, min_power_ratio=0.2, window=5000):
    """
    偵測 df[hip_col] 中有明顯 SIN 波的區段，回傳 (start_idx, end_idx)。
    以傅立葉頻譜能量比判斷，找出主頻能量佔比高的區段。
    Args:
        df: DataFrame
        hip_col: 欄位名稱
        time_col: 時間欄位
        min_period, max_period: 允許的主頻週期範圍 (ms)
        min_power_ratio: 主頻能量佔比門檻
        window: 檢查視窗長度（點數）
    Returns:
        (start_idx, end_idx): 有 SIN 波的區段 index
    """
    hip = df[hip_col].values
    time = df[time_col].values
    n = len(hip)
    window = min(window, n)  # 檢查視窗
    step = window // 10
    best_ratio = 0
    best_range = (0, window)
    for start in range(0, n-window, step):
        seg = hip[start:start+window]
        if np.any(np.isnan(seg)):
            continue
        seg = seg - np.mean(seg)
        fft = np.fft.rfft(seg)
        power = np.abs(fft)**2
        freqs = np.fft.rfftfreq(len(seg), d=(time[1]-time[0])/1000)
        # 主頻
        main_idx = np.argmax(power[1:]) + 1
        main_freq = freqs[main_idx]
        if main_freq == 0:
            continue
        period = 1/main_freq*1000
        if not (min_period <= period <= max_period):
            continue
        main_power = power[main_idx]
        total_power = np.sum(power[1:])
        ratio = main_power / total_power if total_power > 0 else 0
        if ratio > best_ratio and ratio > min_power_ratio:
            best_ratio = ratio
            best_range = (start, start+window)
    return best_range

def safe_savgol_filter(arr, window_length=61, polyorder=3, **kwargs):
    n = len(arr)
    # window_length 不能大於 n，且必須是奇數且 >= polyorder+2
    win = min(window_length, n if n % 2 == 1 else n-1)
    if win < polyorder + 2:
        win = polyorder + 2
        if win % 2 == 0:
            win += 1
    if win > n:
        win = n if n % 2 == 1 else n-1
    if win < 3:
        return arr  # 無法平滑
    return savgol_filter(arr, window_length=win, polyorder=polyorder, **kwargs)

def plot_sin_segment(df, sin_start, sin_end, col, title):
    """
    畫出 df[col]，並用紅線標記 sin_start:sin_end 區段
    """
    plt.figure(figsize=(12,4))
    plt.plot(df['TimeOffset'], df[col], label=f'{col}')
    plt.plot(df['TimeOffset'].iloc[sin_start:sin_end], df[col].iloc[sin_start:sin_end], color='red', linewidth=3, label='sin波擷取區段')
    plt.xlabel('TimeOffset (ms)')
    plt.ylabel(col)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def plot_angle_sin_segment(df, hip_segments, ankle_segments):
    """
    畫圖顯示 df hip 及用 sin 波補值的區段（紅色），ankle 用藍色
    hip_segments, ankle_segments: list of (start, end) index tuples
    """
    plt.figure(figsize=(12,4))
    # plt.plot(df['TimeOffset'], df['knee'], label='Knee angle')
    plt.plot(df['TimeOffset'], df['hip'], label='Hip angle')
    for idx, (start, end) in enumerate(hip_segments):
        plt.plot(df['TimeOffset'].iloc[start:end], df['hip'].iloc[start:end], color='red', linewidth=3, label='Compensated hip angle' if idx==0 else "")
    plt.plot(df['TimeOffset'], df['ankle'], label='Ankle angle')
    for idx, (start, end) in enumerate(ankle_segments):
        plt.plot(df['TimeOffset'].iloc[start:end], df['ankle'].iloc[start:end], color='blue', linewidth=3, label='Compensated ankle angle' if idx==0 else "")
    plt.xlabel('TimeOffset (ms)')
    plt.ylabel('Angle (deg)')
    plt.title('Hip & Ankle angles after signal processing')
    plt.legend(loc="center right")
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def detect_and_fill_nan_segments(df, col, pattern, min_duration, fill_segments_list=None):
    """
    偵測 col 欄位 NaN 區段，超過 min_duration 時用 pattern 補值，並記錄補值區段。
    回傳 (新 df, 補值區段 list)
    """
    df = df.copy()
    nan_mask = df[col].isna().values
    segments = []
    start = None
    for i, v in enumerate(nan_mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > 0:
                segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(nan_mask)))
    fill_segments = []
    for start, end in segments:
        seg_len = end - start
        if seg_len >= min_duration:
            if len(pattern) < 2:
                continue
            pattern_resampled = np.interp(np.linspace(0, len(pattern)-1, seg_len), np.arange(len(pattern)), pattern)
            df.loc[start:end-1, col] = pattern_resampled[:end-start]
            fill_segments.append((start, end))
    if fill_segments_list is not None:
        fill_segments_list.extend(fill_segments)
    return df, fill_segments

def detect_and_fill_fixed_segments(df, col, fixed_value_list, pattern, min_duration, fill_segments_list=None):
    """
    偵測 col 欄位連續出現 fixed_value_list 中任一值的區段，超過 min_duration 時用 pattern 補值，並記錄補值區段。
    回傳 (新 df, 補值區段 list)
    """
    df = df.copy()
    fixed_mask = np.isin(df[col], fixed_value_list)
    segments = []
    start = None
    for i, v in enumerate(fixed_mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > 0:
                segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(fixed_mask)))
    fill_segments = []
    for start, end in segments:
        seg_len = end - start
        if seg_len >= min_duration:
            if len(pattern) < 2:
                continue
            pattern_resampled = np.interp(np.linspace(0, len(pattern)-1, seg_len), np.arange(len(pattern)), pattern)
            df.loc[start:end-1, col] = pattern_resampled[:end-start]
            fill_segments.append((start, end))
    if fill_segments_list is not None:
        fill_segments_list.extend(fill_segments)
    return df, fill_segments
