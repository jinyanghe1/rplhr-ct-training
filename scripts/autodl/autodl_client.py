"""
AutoDL API 客户端
用于控制服务器开关机、查询状态
"""
import requests
import time
from typing import Optional, Dict, Any


class AutoDLClient:
    """AutoDL OpenAPI 客户端"""
    
    def __init__(self, api_token: str):
        """
        初始化客户端
        
        Args:
            api_token: AutoDL API 密钥 (从控制台获取)
        """
        self.token = api_token
        self.base_url = "https://www.autodl.com/api/v1"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
    
    def list_instances(self) -> Dict[str, Any]:
        """获取所有实例列表"""
        resp = requests.get(
            f"{self.base_url}/instances",
            headers=self.headers
        )
        resp.raise_for_status()
        return resp.json()
    
    def get_instance(self, instance_id: str) -> Dict[str, Any]:
        """
        获取实例详情
        
        Args:
            instance_id: 实例 ID (形如: 1234567890abcdef)
        
        Returns:
            实例信息，包含 status (running/stopped/starting 等)
        """
        resp = requests.get(
            f"{self.base_url}/instances/{instance_id}",
            headers=self.headers
        )
        resp.raise_for_status()
        return resp.json()
    
    def power_on(self, instance_id: str) -> Dict[str, Any]:
        """
        开机
        
        Args:
            instance_id: 实例 ID
        
        Returns:
            API 响应
        """
        print(f"🔌 正在开机: {instance_id}")
        resp = requests.post(
            f"{self.base_url}/instances/{instance_id}/power_on",
            headers=self.headers
        )
        resp.raise_for_status()
        return resp.json()
    
    def power_off(self, instance_id: str) -> Dict[str, Any]:
        """
        关机
        
        Args:
            instance_id: 实例 ID
        
        Returns:
            API 响应
        """
        print(f"🔌 正在关机: {instance_id}")
        resp = requests.post(
            f"{self.base_url}/instances/{instance_id}/power_off",
            headers=self.headers
        )
        resp.raise_for_status()
        return resp.json()
    
    def wait_for_status(self, instance_id: str, target_status: str, 
                        timeout: int = 300, poll_interval: int = 10) -> bool:
        """
        等待实例达到目标状态
        
        Args:
            instance_id: 实例 ID
            target_status: 目标状态 (running/stopped)
            timeout: 超时时间(秒)
            poll_interval: 轮询间隔(秒)
        
        Returns:
            是否成功达到目标状态
        
        Raises:
            TimeoutError: 超时
        """
        print(f"⏳ 等待实例 {instance_id} 达到状态: {target_status}")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            info = self.get_instance(instance_id)
            current_status = info.get("status")
            
            print(f"   当前状态: {current_status} ({int(time.time() - start_time)}s)")
            
            if current_status == target_status:
                print(f"✅ 实例已达到目标状态: {target_status}")
                return True
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"等待实例状态超时 ({timeout}s)")
    
    def wait_for_running(self, instance_id: str, timeout: int = 300) -> bool:
        """等待实例开机完成"""
        return self.wait_for_status(instance_id, "running", timeout)
    
    def wait_for_stopped(self, instance_id: str, timeout: int = 60) -> bool:
        """等待实例关机完成"""
        return self.wait_for_status(instance_id, "stopped", timeout)
    
    def get_gpu_info(self, instance_id: str) -> Dict[str, Any]:
        """获取实例 GPU 信息"""
        info = self.get_instance(instance_id)
        return info.get("gpu", {})


if __name__ == "__main__":
    # 测试代码
    import os
    
    token = os.getenv("AUTODL_TOKEN", "your_token_here")
    client = AutoDLClient(token)
    
    # 列出所有实例
    instances = client.list_instances()
    print("实例列表:", instances)
