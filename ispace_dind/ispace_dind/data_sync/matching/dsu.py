class DSU:
    
    def __init__(self):
        self.unique_to_local = {}
        self.local_to_uniques = {}
        
    def add_unique_id(self, unique_id, local_id):
        #uniqueIDが既に存在する場合は失敗
        if unique_id in self.unique_to_local:
            return False
        #local_idが既に同様のhostnameのuniqueIDに紐づいている場合は失敗
        unique_id_list = self.local_to_uniques.get(local_id, [])
        hostname = unique_id.split('_')[0]
        if any(uid.startswith(hostname) for uid in unique_id_list):
            return False
        self.unique_to_local[unique_id] = local_id    
        self.local_to_uniques[local_id] = unique_id_list + [unique_id]
        return True
        
    def has_unique_id(self, unique_id):
        return unique_id in self.unique_to_local
    
    def get_local_id_from_unique(self, unique_id):
        return self.unique_to_local.get(unique_id)
    
    def get_unique_ids_from_local(self, local_id):
        return self.local_to_uniques.get(local_id, [])