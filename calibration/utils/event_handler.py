
class EventHandler:
    
    def __init__(self):
        self.listeners = {}  # イベント名をキーにリスナー（コールバック）のリストを保持

    def add_listener(self, event_name, callback):
        """ 指定のイベント名に対してリスナーを登録 """
        if event_name not in self.listeners:
            self.listeners[event_name] = []
        self.listeners[event_name].append(callback)

    def remove_listener(self, event_name, callback):
        """ 指定のイベント名からリスナーを削除 """
        if event_name in self.listeners:
            self.listeners[event_name].remove(callback)
            if not self.listeners[event_name]:
                del self.listeners[event_name]

    def emit(self, event_name, *args, **kwargs):
        """ 指定のイベントを発火（emit）し、登録済みのリスナーに引数を渡して呼び出す """
        if event_name in self.listeners:
            for callback in self.listeners[event_name]:
                callback(*args, **kwargs)

class Event:

    IMAGE_GET_EVENT = "image_get_event"
    DATA_SYNC_EVENT = "data_sync_event"
    YOLO_EVENT = "yolo_event"
    CLICKED_EVENT = "clicked_event"
