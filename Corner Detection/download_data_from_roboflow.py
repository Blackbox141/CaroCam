
from roboflow import Roboflow

rf = Roboflow(api_key="rbM07YOtK6iupmE214E4")
project = rf.workspace("camocam").project("other-data-for-carocam")
version = project.version(5)
dataset = version.download("yolov8")
