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
import matplotlib.pyplot as plt

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
    K = 180 / (2**14 - 1) * 3.6
    if data_high != 0 and data_low != 0: 
        raw_value = (data_high << 8) | data_low
        phi = 460 - raw_value * K  # angle multiplied by 2 to be a real degreeWSS
        if phi < 200:
            return phi
    else:
        return None

def decode_phi_knee(data_high, data_low):
    # 14-bit raw data, angle range [0°,180°] or [-90°,90°]
    K = 180 / (2**14 - 1) * 3.6
    raw_value = (data_high << 8) | data_low
    phi = raw_value * K - 95 # angle multiplied by 2 to be a real degreeWSS
    return phi

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

def decode_grf(byte_high, byte_low):
    raw = (byte_high << 8) | byte_low
    return raw / 65536 * 1000 / 20
def decode_GRF(byte_high, byte_low):
    # Step 1: Combine bytes to get raw ADC value
    raw = (byte_high << 8) | byte_low  # 10-bit expected: max 1024
    
    # Step 2: Convert to voltage
    voltage = raw * 5.0 / 1024.0

    # Step 3: Voltage cap check
    if voltage >= 5.0:
        return 0.0

    # Step 4: Estimate weight (g) from voltage (placeholder function)
    # TODO: Replace with calibrated curve or function
    weight = voltage_to_weight(voltage)

    # Step 5: Convert to Newton
    force = weight * 9.81 / 1000.0
    return force

def voltage_to_weight(voltage):
    # 🔧 Linear placeholder: assume 0V → 0g, 5V → 1000g
    # You can replace this with a nonlinear curve or lookup table
    return voltage * 400 # 5V → 1000g => 200g/V

# =============================================================
# 3. Gait Phase Detection
# =============================================================

# def detect_gait_phase(knee, hip, ankle, grf):
#     if grf > 20:  # Stance Phase
#         if knee >= 135 and ankle <= 95 and hip <= 100:
#             return 'Initial Contact'
#         elif 110 <= knee < 135 and ankle <= 100 and 100 < hip <= 115:
#             return 'Loading Response'
#         elif 105 <= knee <= 120 and 100 < ankle <= 110 and 110 < hip <= 125:
#             return 'Mid Stance'
#         elif 100 <= knee < 110 and ankle > 110 and hip > 125:
#             return 'Terminal Stance'
#         else:
#             return 'Pre-Swing'

#     else:  # Swing Phase
#         if knee > 150 and hip > 130:
#             return 'Initial Swing'
#         elif 135 < knee <= 150 and hip >= 135:
#             return 'Mid Swing'
#         elif 115 <= knee <= 135 and 125 <= hip <= 135:
#             return 'Terminal Swing'
#         else:
#             return 'Unknown (Swing)'

def detect_gait_phase(knee, hip, grf, imu_pitch, imu_roll):
    if grf > 18:  # Stance phase
        if knee > 170 and hip > 145 and imu_pitch < -68:
            return 'Initial Contact'
        elif 160 < knee <= 170 and 135 < hip <= 145 and -70 < imu_pitch < -65:
            return 'Loading Response'
        elif 150 < knee <= 160 and 130 < hip <= 135 and -67 < imu_pitch < -62:
            return 'Mid Stance'
        elif knee < 150 and hip < 130 and imu_pitch > -65:
            return 'Terminal Stance'
        else:
            return 'Pre-Swing'
    else:  # Swing phase
        if knee < 140 and hip < 130 and imu_pitch > -67:
            return 'Initial Swing'
        elif 140 <= knee <= 160 and 130 <= hip <= 140 and -70 < imu_pitch < -65:
            return 'Mid Swing'
        elif knee > 160 and hip > 140 and imu_pitch < -70:
            return 'Terminal Swing'
        else:
            return 'Unknown (Swing)'
        
def detect_gait_phase_row(row):
    # 若沒有 grf 欄位，預設為 0
    grf = row["grf"] if "grf" in row else 30
    femur_pitch = row["femur_pitch"] if "femur_pitch" in row else 0
    femur_roll = row["femur_roll"] if "femur_roll" in row else 0
    femur_yaw = row["femur_yaw"] if "femur_yaw" in row else 0
    return detect_gait_phase(row["knee"], row["hip"], grf, femur_pitch, femur_roll)

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

# def plot_encoder(data, name):
#     if name not in encoder_id_label:
#         print(f"[Error] Unknown encoder name: {name}")
#         return
#     can_id = encoder_id_label[name]
#     if can_id not in data or not data[can_id]:
#         print(f"[Warning] No data for {name} (CAN ID {can_id})")
#         return
#     df = pd.DataFrame(data[can_id])
#     plt.figure()
#     plt.plot(df["TimeOffset"], df["Angle"], marker='o')
#     plt.title(f"{name} - ID {can_id}")
#     plt.xlabel("Time Offset (ms)")
#     plt.ylabel("Angle (degrees)")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

def plot_encoder(data, name1, name2=None):
    if name1 not in encoder_id_label:
        print(f"[Error] Unknown encoder name: {name1}")
        return
    can_id1 = encoder_id_label[name1]
    if can_id1 not in data or not data[can_id1]:
        print(f"[Warning] No data for {name1} (CAN ID {can_id1})")
        return

    df1 = pd.DataFrame(data[can_id1])
    plt.figure()
    plt.plot(df1["TimeOffset"], df1["Angle"], marker='o', label=name1)

    if name2:
        if name2 not in encoder_id_label:
            print(f"[Error] Unknown encoder name: {name2}")
        else:
            can_id2 = encoder_id_label[name2]
            if can_id2 not in data or not data[can_id2]:
                print(f"[Warning] No data for {name2} (CAN ID {can_id2})")
            else:
                df2 = pd.DataFrame(data[can_id2])
                plt.plot(df2["TimeOffset"], df2["Angle"], marker='x', label=name2)

    plt.title(f"Left Knee Angle vs Left Hip Angle")
    plt.xlabel("Time Offset (ms)")
    plt.ylabel("Angle (degrees)")
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
        plt.xlabel("Time Offset (ms)")
        plt.ylabel("Angle (degrees)")
        plt.grid(True)
        plt.tight_layout()
    plt.show()

def plot_grf(data, name):
    if name not in grf_id_label:
        print(f"[Error] Unknown GRF name: {name}")
        return
    can_id, channel = grf_id_label[name]
    if can_id not in data or channel not in data[can_id] or not data[can_id][channel]:
        print(f"[Warning] No data for {name} (CAN ID {can_id}, {channel})")
        return
    df = pd.DataFrame(data[can_id][channel])
    plt.figure()
    plt.plot(df["TimeOffset"], df["Value"], marker='.')
    plt.title(f"{name} - ID {can_id} / {channel}")
    plt.xlabel("Time Offset (ms)")
    plt.ylabel("Force Value (N)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()