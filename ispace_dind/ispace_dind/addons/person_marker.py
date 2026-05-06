from ispace_dind.addons.addon_base import AddonBase, addon
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from ispace_dind.utils.event_handler import Event

@addon
class PersonMarker(AddonBase):

    def __init__(self, node):
        self.node = node

    def register(self):
        self.marker = None
        self.marker_pub = self.node.create_publisher(MarkerArray, 'marker_sync2', 10)
        self.node.event_handler.add_listener(Event.DATA_SYNC_EVENT, self.publish_marker)

    def publish_marker(self, data):
        marker_array = MarkerArray()
        markers = []
        for tracker in data['update_trackers']:
            if tracker.get_local_id() < 0:
                continue
            cylinder, text = self.get_person_marker(tracker.ekf.ekf.x, tracker.get_local_id())
            markers.append(cylinder)
            markers.append(text)
        marker_array.markers = markers
        self.marker_pub.publish(marker_array)

    def get_person_marker(self, position, id):
        cylinder = self.get_cylinder_marker(id*2, (position[0], position[1], 0.1), (1.0,0.0,0.0,0.0), (1.0,1.0,1.0), (1.0,0.0,1.0,0.0))
        text = self.get_text_marker(str(id), id*2+1, (position[0],position[1],0.3), (1.0,0.0,0.0,0.0), (1.0,1.0,1.0), (1.0,1.0,1.0,1.0))
        return cylinder, text

    def get_cylinder_marker(self, id, posision, orientation, scale, color):
        marker = self.__create_base_marker(id, posision, orientation, scale, color)
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.scale.z = 0.2
        return marker

    def get_text_marker(self, text, id, posision, orientation, scale, color):
        marker = self.__create_base_marker(id, posision, orientation, scale, color)
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.text = text
        marker = self.__set_marker_color(marker, 1.0, 1.0, 1.0, 1.0)
        marker = self.__set_marker_scale(marker, 0.2, 0.2, 0.2)
        return marker

    def __create_base_marker(self, id=0, posision=(0.0,0.0,0.1), orientation=(1.0,0.0,0.0,0.0), scale=(0.1,0.1,0.2), color=(1.0,1.0,1.0,1.0)):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.ns = "isp"
        marker.id = id
        marker.pose.position.x = posision[0]
        marker.pose.position.y = posision[1]
        marker.pose.position.z = posision[2]
        marker.pose.orientation.w = 1.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = posision[2]
        marker.color.a = 0.5
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.lifetime.sec = 1
        return marker
    
    def __set_marker_color(self, marker, a, r, g, b):
        marker.color.a = a
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        return marker
    
    def __set_marker_scale(self, marker, x, y, z):
        marker.scale.x = x
        marker.scale.y = y
        marker.scale.z = z
        return marker
