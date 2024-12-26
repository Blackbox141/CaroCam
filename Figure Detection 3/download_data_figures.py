from roboflow import Roboflow

rf = Roboflow(api_key="tBrWxTCEFIWMdWsL1DiF")
project = rf.workspace("carocam").project("carocam-project")
version = project.version(8)
dataset = version.download("yolov8")
