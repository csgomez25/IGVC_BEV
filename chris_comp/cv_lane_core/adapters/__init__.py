"""
Live-capture adapters — the ONLY place ROS / hardware deps are allowed.

`lane_cv` stays import-clean (numpy + opencv only); these adapters wrap it for a
specific input source and are imported explicitly by their entry-point scripts,
never by the core. Nothing here is imported at package load, so
`import lane_cv` never drags in rclpy.

  usb_cam.py   plain OpenCV VideoCapture (USB / UVC webcam or a video file)
  ros2_zed.py  rclpy node for the ZED X wrapper topics (1 or 3 cameras)

See adapters/README.md for run commands.
"""
