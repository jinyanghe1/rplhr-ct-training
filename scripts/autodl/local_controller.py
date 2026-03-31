"""
本地控制器 - 连接 AutoDL 执行完整 MLOps 流程
在本地运行，控制 AutoDL 服务器开关机、训练、结果下载
"""
import os
import sys
import time
import json
from typing import Dict, Any, Optional
from datetime import datetime

import paramiko
from paramiko import SSHClient, AutoAddPolicy

# 添加脚本目录到路径
sys.path.insert(0, os.path.dirname(__file__))
from autodl_client import AutoDLClient


class RPLHRController:
    """
    RPLHR-CT AutoDL 控制器
    
    完整流程: 开机 → SSH连接 → git pull → 训练 → 评估 → 下载结果 → 关机
    """
    
    def __init__(self, api_token: str, instance_id: str, ssh_config: Dict[str, Any]):
        """
        初始化控制器
        
        Args:
            api_token: AutoDL API Token
            instance_id: AutoDL 实例 ID
            ssh_config: SSH 连接配置 {
                "host": "xxx.autodl.com",
                "port": 22,
                "username": "root",
                "password": "your_password"
            }
        """
        self.autodl = AutoDLClient(api_token)
        self.instance_id = instance_id
        self.ssh_config = ssh_config
        self.ssh: Optional[SSHClient] = None
        self.sftp = None
        
        # 本地报告目录
        self.local_report_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "autodl_reports"
        )
        os.makedirs(self.local_report_dir, exist_ok=True)
    
    def power_on_and_wait(self, timeout: int = 300) -> bool:
        """
        开机并等待启动完成
        
        Returns:
            是否成功启动
        """
        try:
            # 检查当前状态
            info = self.autodl.get_instance(self.instance_id)
            current_status = info.get("status")
            
            if current_status == "running":
                print("✅ 实例已在运行中")
                return True
            
            # 开机
            self.autodl.power_on(self.instance_id)
            
            # 等待启动
            self.autodl.wait_for_running(self.instance_id, timeout)
            
            # 等待 SSH 服务启动 (需要额外时间)
            print("⏳ 等待 SSH 服务启动...")
            time.sleep(30)
            
            return True
            
        except Exception as e:
            print(f"❌ 开机失败: {e}")
            return False
    
    def connect_ssh(self, max_retries: int = 5, retry_delay: int = 10) -> bool:
        """
        建立 SSH 连接
        
        Returns:
            是否成功连接
        """
        print(f"🔌 连接 SSH: {self.ssh_config['host']}:{self.ssh_config['port']}")
        
        for attempt in range(max_retries):
            try:
                self.ssh = SSHClient()
                self.ssh.set_missing_host_key_policy(AutoAddPolicy())
                
                self.ssh.connect(
                    hostname=self.ssh_config["host"],
                    port=self.ssh_config["port"],
                    username=self.ssh_config["username"],
                    password=self.ssh_config["password"],
                    timeout=30
                )
                
                print("✅ SSH 连接成功")
                return True
                
            except Exception as e:
                print(f"   尝试 {attempt + 1}/{max_retries} 失败: {e}")
                if attempt < max_retries - 1:
                    print(f"   等待 {retry_delay}s 后重试...")
                    time.sleep(retry_delay)
        
        print("❌ SSH 连接失败")
        return False
    
    def execute_training(self, config_file: str = "config/ratio4.txt") -> Dict[str, Any]:
        """
        在 AutoDL 上执行训练流水线
        
        Args:
            config_file: 训练配置文件
        
        Returns:
            执行结果
        """
        print(f"🚀 在 AutoDL 上执行训练 (config: {config_file})")
        
        # 执行 autodl_train.py
        repo_path = "/root/rplhr-ct-training"  # AutoDL 上的路径
        cmd = f"cd {repo_path} && python autodl_train.py {config_file}"
        
        print(f"   执行命令: {cmd}")
        
        stdin, stdout, stderr = self.ssh.exec_command(cmd, timeout=3600*4)  # 4小时超时
        
        # 实时输出
        output_lines = []
        for line in stdout:
            line = line.rstrip()
            print(f"   [AutoDL] {line}")
            output_lines.append(line)
        
        output = "\n".join(output_lines)
        error = stderr.read().decode()
        
        if error:
            print(f"⚠️ 警告输出:\n{error}")
        
        # 解析结果 JSON
        results = {}
        if "RESULTS_JSON:" in output:
            try:
                json_start = output.index("RESULTS_JSON:") + len("RESULTS_JSON:")
                json_str = output[json_start:].strip()
                results = json.loads(json_str)
            except Exception as e:
                print(f"⚠️ 解析结果 JSON 失败: {e}")
        
        return results
    
    def download_results(self) -> bool:
        """
        下载训练结果到本地
        
        Returns:
            是否成功下载
        """
        print("📥 下载训练结果...")
        
        try:
            self.sftp = self.ssh.open_sftp()
            
            remote_base = "/root/rplhr-ct-training"
            
            # 下载最新报告
            remote_report = f"{remote_base}/reports/latest_report.md"
            local_report = os.path.join(
                self.local_report_dir, 
                f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
            
            try:
                self.sftp.get(remote_report, local_report)
                print(f"✅ 报告已下载: {local_report}")
            except FileNotFoundError:
                print("⚠️ 远程报告文件不存在")
            
            # 下载指标历史
            remote_history = f"{remote_base}/metrics_history.json"
            local_history = os.path.join(self.local_report_dir, "metrics_history.json")
            
            try:
                self.sftp.get(remote_history, local_history)
                print(f"✅ 指标历史已下载: {local_history}")
            except FileNotFoundError:
                print("⚠️ 远程指标历史不存在")
            
            self.sftp.close()
            return True
            
        except Exception as e:
            print(f"❌ 下载结果失败: {e}")
            return False
    
    def shutdown(self) -> bool:
        """
        关机
        
        Returns:
            是否成功关机
        """
        print("🔌 关闭 AutoDL 实例...")
        
        try:
            # 关闭 SSH 连接
            if self.ssh:
                self.ssh.close()
                print("   SSH 连接已关闭")
            
            # 关机
            self.autodl.power_off(self.instance_id)
            
            print("✅ 关机命令已发送")
            return True
            
        except Exception as e:
            print(f"❌ 关机失败: {e}")
            return False
    
    def run_pipeline(self, config_file: str = "config/ratio4.txt") -> Dict[str, Any]:
        """
        执行完整 MLOps 流程
        
        Args:
            config_file: 训练配置文件
        
        Returns:
            执行结果汇总
        """
        print("=" * 70)
        print("🚀 RPLHR-CT AutoDL MLOps 流程开始")
        print("=" * 70)
        print(f"实例 ID: {self.instance_id}")
        print(f"配置文件: {config_file}")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        results = {
            "success": False,
            "instance_id": self.instance_id,
            "config": config_file,
            "start_time": datetime.now().isoformat(),
            "training_results": {},
            "errors": []
        }
        
        try:
            # 1. 开机
            if not self.power_on_and_wait():
                results["errors"].append("开机失败")
                return results
            
            # 2. SSH 连接
            if not self.connect_ssh():
                results["errors"].append("SSH连接失败")
                self.shutdown()
                return results
            
            # 3. 执行训练
            training_results = self.execute_training(config_file)
            results["training_results"] = training_results
            
            if training_results.get("success"):
                print("✅ 训练成功完成")
            else:
                print("⚠️ 训练可能有问题")
            
            # 4. 下载结果
            self.download_results()
            
            results["success"] = True
            
        except Exception as e:
            error_msg = f"流程执行出错: {str(e)}"
            print(f"❌ {error_msg}")
            results["errors"].append(error_msg)
            
        finally:
            # 5. 关机
            self.shutdown()
            
            results["end_time"] = datetime.now().isoformat()
            
            print("=" * 70)
            print("🏁 MLOps 流程结束")
            print(f"成功: {results['success']}")
            print(f"耗时: {self._format_duration(results.get('start_time'), results.get('end_time'))}")
            print("=" * 70)
        
        return results
    
    def _format_duration(self, start: str, end: str) -> str:
        """格式化时间差"""
        try:
            from datetime import datetime
            t1 = datetime.fromisoformat(start)
            t2 = datetime.fromisoformat(end)
            delta = t2 - t1
            hours, remainder = divmod(delta.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours}h {minutes}m {seconds}s"
        except:
            return "N/A"


def load_config() -> Dict[str, Any]:
    """从环境变量或配置文件加载配置"""
    config = {
        "api_token": os.getenv("AUTODL_TOKEN", ""),
        "instance_id": os.getenv("AUTODL_INSTANCE_ID", ""),
        "ssh_host": os.getenv("AUTODL_SSH_HOST", ""),
        "ssh_port": int(os.getenv("AUTODL_SSH_PORT", "22")),
        "ssh_username": os.getenv("AUTODL_SSH_USER", "root"),
        "ssh_password": os.getenv("AUTODL_SSH_PASS", "")
    }
    
    # 尝试从文件加载
    config_file = os.path.expanduser("~/.autodl_config.json")
    if os.path.exists(config_file):
        with open(config_file) as f:
            file_config = json.load(f)
            config.update(file_config)
    
    return config


if __name__ == "__main__":
    # 加载配置
    config = load_config()
    
    if not config["api_token"]:
        print("❌ 错误: 未设置 AUTODL_TOKEN")
        print("   请设置环境变量或创建 ~/.autodl_config.json")
        print()
        print("环境变量:")
        print("   export AUTODL_TOKEN=your_token")
        print("   export AUTODL_INSTANCE_ID=your_instance_id")
        print("   export AUTODL_SSH_HOST=xxx.autodl.com")
        print("   export AUTODL_SSH_PASS=your_password")
        sys.exit(1)
    
    ssh_config = {
        "host": config["ssh_host"],
        "port": config["ssh_port"],
        "username": config["ssh_username"],
        "password": config["ssh_password"]
    }
    
    controller = RPLHRController(
        api_token=config["api_token"],
        instance_id=config["instance_id"],
        ssh_config=ssh_config
    )
    
    # 从命令行获取配置文件
    train_config = sys.argv[1] if len(sys.argv) > 1 else "config/ratio4.txt"
    
    # 执行完整流程
    results = controller.run_pipeline(train_config)
    
    # 保存结果
    result_file = os.path.join(
        controller.local_report_dir,
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 运行结果已保存: {result_file}")
