from abc import ABC, abstractmethod
from typing import Type, Any
from functools import wraps

def addon(cls: Type) -> Type:
    """
    アドオンクラスを登録するためのデコレータ
    使用例:
    @addon
    class MyAddon(AddonBase):
        def register(self):
            pass
    """
    @wraps(cls)
    def register_addon(node: Any) -> Any:
        instance = cls(node)
        instance.register()
        return instance
    
    # クラスにregister_addon関数を追加
    cls.register_addon = staticmethod(register_addon)
    return cls

class AddonBase(ABC):
    """
    アドオンの基底クラス
    すべてのアドオンはこのクラスを継承する必要があります
    """

    def __init__(self, node):
        """
        Args:
            node: ROS2ノードインスタンス
        """
        self.node = node

    @abstractmethod
    def register(self):
        """
        アドオンの初期化と登録を行うメソッド
        このメソッドはサブクラスで実装する必要があります
        """
        raise NotImplementedError("register()メソッドを実装してください")
    