
class IDManager:
    
    def __init__(self):
        self.id_counter = 0
        self.pre_id_counter = 0
        
    def get_next_id(self):
        self.id_counter += 1
        return self.id_counter
    
    def get_next_pre_id(self):
        self.pre_id_counter -= 1
        return self.pre_id_counter