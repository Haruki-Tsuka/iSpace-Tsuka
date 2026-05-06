import pyrealsense2 as rs
import numpy as np

class RealSenseManager:

    def __init__(self, width=640, height=480, fps=30):
        self.width, self.height = width, height
        self.align, self.config, self.pipeline, self.profile = realsense_setting(width, height, fps)
        self.color_intr = rs.video_stream_profile(self.profile.get_stream(rs.stream.color)).get_intrinsics()
        self.decimate, self.spatial, self.depth_to_disparity, self.disparity_to_depth = get_filters()

    def __get_filtered_depth_frame(self, depth_frame):
        filter_frame = self.decimate.process(depth_frame)
        filter_frame = self.depth_to_disparity.process(filter_frame)
        filter_frame = self.spatial.process(filter_frame)
        filter_frame = self.disparity_to_depth.process(filter_frame)
        return filter_frame.as_depth_frame()
    
    def is_in_frame(self, pixel_x, pixel_y):
        return (pixel_x >= 0 and pixel_x < self.width and pixel_y >= 0 and pixel_y < self.height)
    
    def update(self):
        while True:
            frames = self.pipeline.wait_for_frames(10000)
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            self.color_frame, self.depth_frame = color_frame, self.__get_filtered_depth_frame(depth_frame)
            break
    
    def close(self):
        self.pipeline.stop()

    def get_colorizing_depth(self):
        colorizer = rs.colorizer()
        return np.asanyarray(colorizer.colorize(self.depth_frame).get_data())
    
    def get_depth_numpy(self):
        return np.array(self.depth_frame.get_data(), dtype=np.float32)/1000.0
    
    def get_distance(self, pixel_x, pixel_y):
        if not self.is_in_frame(pixel_x, pixel_y):
            return 0
        return self.depth_frame.get_distance(pixel_x,pixel_y)

    def get_3d_coordinate(self, pixel_x, pixel_y, depth=None):
        if not self.is_in_frame(pixel_x, pixel_y):
            return None
        if not depth:
            depth = self.depth_frame.get_distance(pixel_x,pixel_y)
        if depth == 0:
            return None
        return rs.rs2_deproject_pixel_to_point(self.color_intr , [pixel_x,pixel_y], depth)
    
    def get_img(self):
        return np.asanyarray(self.color_frame.get_data())

def realsense_setting(width, height, fps):
    align = rs.align(rs.stream.color)
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    return (align, config, pipeline, profile)

def get_filters():
    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 1)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 1)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.25)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    # disparity
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    return (decimate, spatial, depth_to_disparity, disparity_to_depth)

