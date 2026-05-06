int16 track_id
int8 state  # 0:updated, 1:predicted, 2:observe-only

float32 visual_conf
float32 assosiate_conf
geometry_msgs/Point coord
geometry_msgs/Point ray
string mapping_json

string data