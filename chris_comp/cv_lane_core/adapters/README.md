# Live-capture adapters

Thin wrappers that feed real camera frames into `lane_cv`. The core stays
ROS-free; everything hardware-specific lives here.

| Adapter | Source | Deps |
|---|---|---|
| `usb_cam.py` | any USB/UVC webcam or a video file | opencv only |
| `ros2_zed.py` | ZED X via the ZED ROS 2 wrapper | rclpy, cv_bridge, sensor_msgs |

---

## (a) Random USB camera — no ROS

```bash
# default profile, webcam index 0
python adapters/usb_cam.py --device 0 --config configs/default.yaml

# a specific device node, or a recorded clip
python adapters/usb_cam.py --device /dev/video2
python adapters/usb_cam.py --device clip.mp4 --out _artifacts/usb --no-display
```

Shows a 3-up panel (overlay | candidate | lane_mask) with live FPS.
Keys: `q` quit · `r` reset state · `space` pause · `s` save panel.

If it can't open the device, list what's present: `ls /dev/video*`. Tune the
profile live against the same camera with `python tools/tune.py --input <a saved frame>`.

## (b) ZED X — ROS 2

Subscribes the ZED wrapper's rectified RGB and republishes an overlay + mono8
lane mask per camera. Topic paths default to the **v5.x** convention from the
references (`references/parsa_igvc/CLAUDE.md`).

Prereqs: a sourced ROS 2 Humble env with `cv_bridge`, and the ZED wrapper
already publishing (`sensors.launch.py enable_zed_front:=true` on Parsa's side).

```bash
# just the front camera
python3 adapters/ros2_zed.py --cameras front

# all three (serial-pinned namespaces left / front / right)
python3 adapters/ros2_zed.py --cameras front left right
```

Published per camera:

| Topic | Type | Notes |
|---|---|---|
| `/lane/<cam>/mask` | `sensor_msgs/Image` mono8 | 0/255 confirmed lane pixels; header copied from source |
| `/lane/<cam>/overlay` | `sensor_msgs/Image` bgr8 | debug overlay (disable with `-p publish_overlay:=false`) |

Verify:

```bash
ros2 topic hz /lane/front/mask
ros2 run rqt_image_view rqt_image_view /lane/front/overlay
```

### Topic / camera mapping (from the references)

| Camera | Namespace | RGB topic (v5) |
|---|---|---|
| Left  | `zed_left`  | `/zed_left/zed_node/rgb/color/rect/image`  |
| Front | `zed_front` | `/zed_front/zed_node/rgb/color/rect/image` |
| Right | `zed_right` | `/zed_right/zed_node/rgb/color/rect/image` |

Override for a v4 bringup (path differs — see the references' "ZED v5 topic
names differ from v4" note):

```bash
python3 adapters/ros2_zed.py --cameras front --ros-args \
  -p rgb_topic_template:='/zed_{cam}/zed_node/rgb/image_rect_color'
```

Other overridable params: `config` (profile YAML), `overlay_topic_template`,
`mask_topic_template`, `publish_overlay`.

### Notes

- **One detector per camera** — each keeps independent adaptive-floor + temporal
  state, so one camera's exposure swing can't perturb another.
- This adapter is a 2D per-camera test. Fusing the three masks into one top-down
  BEV grid (using the cameras' intrinsics + TF/mount extrinsics) is **Phase 3**
  in [../TODO.md](../TODO.md) — not done here.
- A camera that isn't publishing just stays silent; the node doesn't crash.
- For per-camera profiles (different ROI/HSV per mount), give each camera its own
  YAML — full multi-camera config tree is **Phase 2**.
