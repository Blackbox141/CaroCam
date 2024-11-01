from roboflow import Roboflow

rf = Roboflow(api_key="wdhI2C8De0NMErp39t4B")
project = rf.workspace("chessclock").project("chess-clock")
version = project.version(1)
dataset = version.download("yolov8")
