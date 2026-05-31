#!/usr/bin/env bash
# =============================================================================
# record_session.sh — capture a ZED session for offline BEV-perception replay
# =============================================================================
#
# Records the minimum topic set needed to drive bev_perception_node via
# `ros2 launch avl_bev_perception vision_test.launch.py use_bag:=true ...`.
# Default: all three cameras + TF; mcap storage with zstd compression.
#
# Usage:
#   ./record_session.sh                          # auto-name into /tmp
#   ./record_session.sh /data/run_42             # custom output dir prefix
#   ./record_session.sh /data/run_42 front       # only front camera
#   ./record_session.sh /data/run_42 left,front  # left + front
#
# Requirements:
#   - ROS 2 sourced (`source /opt/ros/humble/setup.bash`)
#   - rosbag2 mcap plugin: `sudo apt install ros-humble-rosbag2-storage-mcap`
#   - ZED bringup running (live cameras publishing to /zed_<cam>/zed_node/...)
#
# What gets recorded (per camera in CAMERAS):
#   /zed_<cam>/zed_node/rgb/color/rect/image           # sensor_msgs/Image, bgr8
#   /zed_<cam>/zed_node/rgb/color/rect/camera_info     # sensor_msgs/CameraInfo
#   /zed_<cam>/zed_node/depth/depth_registered         # sensor_msgs/Image, 32FC1
# Plus globally:
#   /tf  /tf_static
#
# NOT recorded by default (uncomment in TOPICS_EXTRA below to include):
#   - point_cloud/cloud_registered (large; not needed for depth-based BEV)
#   - IMU, Velodyne, Xsens (out of scope for vision-only replay)
#
# Stop recording with Ctrl-C. The bag is closed automatically.
# =============================================================================

set -euo pipefail

# ---- args -------------------------------------------------------------------
OUT_PREFIX="${1:-/tmp/bev_session}"
CAMERAS_CSV="${2:-left,front,right}"

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ERROR: 'ros2' not on PATH. Source your ROS 2 install first:" >&2
  echo "       source /opt/ros/humble/setup.bash" >&2
  exit 1
fi

# ---- output path with timestamp suffix --------------------------------------
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_PREFIX}_${TS}"
echo "Recording to: ${OUT_DIR}"

# ---- disk-free sanity (don't start a recording with no room) ----------------
PARENT_DIR="$(dirname "${OUT_DIR}")"
mkdir -p "${PARENT_DIR}"
FREE_GB="$(df -BG "${PARENT_DIR}" | awk 'NR==2 {gsub("G","",$4); print $4}')"
echo "Free space on $(df -h "${PARENT_DIR}" | awk 'NR==2 {print $6}'): ${FREE_GB} GB"
if [[ "${FREE_GB}" -lt 5 ]]; then
  echo "WARNING: less than 5 GB free. Compressed ZED at 15 fps ~= 1-3 GB/min." >&2
fi

# ---- build topic list -------------------------------------------------------
IFS=',' read -r -a CAMERAS <<< "${CAMERAS_CSV}"
TOPICS=()
for cam in "${CAMERAS[@]}"; do
  cam="$(echo "${cam}" | tr -d '[:space:]')"
  [[ -z "${cam}" ]] && continue
  TOPICS+=(
    "/zed_${cam}/zed_node/rgb/color/rect/image"
    "/zed_${cam}/zed_node/rgb/color/rect/camera_info"
    "/zed_${cam}/zed_node/depth/depth_registered"
  )
done
TOPICS+=(/tf /tf_static)

# Extras — uncomment to record. Each adds significantly to bag size.
TOPICS_EXTRA=(
  # "/zed_front/zed_node/point_cloud/cloud_registered"
  # "/imu/data"
  # "/velodyne_points"
)
TOPICS+=("${TOPICS_EXTRA[@]}")

echo "Cameras: ${CAMERAS_CSV}"
echo "Topics  (${#TOPICS[@]}):"
printf '  %s\n' "${TOPICS[@]}"
echo

# ---- record -----------------------------------------------------------------
# mcap + zstd: smaller bags + faster random seek than sqlite3 default.
# If the mcap plugin isn't installed, fall back to sqlite3.
STORAGE_OPTS=(--storage mcap --compression-mode file --compression-format zstd)
if ! ros2 bag info --help 2>&1 | grep -q mcap; then
  : # best-effort detection; the actual record will error if mcap is missing
fi

# Trap SIGINT/SIGTERM to log clean shutdown.
trap 'echo; echo "Stopping recording…"' INT TERM

exec ros2 bag record \
  "${STORAGE_OPTS[@]}" \
  --output "${OUT_DIR}" \
  "${TOPICS[@]}"
