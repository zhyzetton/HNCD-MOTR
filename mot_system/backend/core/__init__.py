# MOT System Core Module
# 延迟导入以避免循环依赖

def get_proposal_generator():
    from .detector import ProposalGenerator
    return ProposalGenerator

def get_tracker():
    from .tracker import HNCDTracker
    return HNCDTracker

def get_anomaly_detector():
    from .anomaly_detector import AnomalyDetector
    return AnomalyDetector

def get_video_processor():
    from .pipeline import VideoProcessor
    return VideoProcessor
