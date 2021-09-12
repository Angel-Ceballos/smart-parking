class Car:
    def __init__(self, img, time) -> None:
        self.img = img 
        self.detection_time = time
    
    def set_ocr_info(self, ocr_json):
        self.ocr_info = ocr_json
    
    def get_ocr_info(self):
        return self.ocr_info