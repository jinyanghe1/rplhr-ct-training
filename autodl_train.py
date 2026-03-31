"""
AutoDL 训练流水线
在 AutoDL 服务器上执行：git pull → 训练 → 评估 → 存档
"""
import os
import sys
import json
import subprocess
import re
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# 添加 code 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))


class TrainingPipeline:
    """RPLHR-CT 训练流水线"""
    
    def __init__(self, config_file: str = "config/default.txt"):
        """
        初始化训练流水线
        
        Args:
            config_file: 配置文件路径 (相对于 code/ 目录)
        """
        self.config_file = config_file
        self.code_dir = os.path.join(os.path.dirname(__file__), '..', 'code')
        self.root_dir = os.path.dirname(__file__)
        self.metrics_history_file = os.path.join(self.root_dir, "metrics_history.json")
        
    def git_sync(self) -> str:
        """
        同步 GitHub 最新代码
        
        Returns:
            当前 git commit hash
        """
        print("🔄 同步 GitHub 最新代码...")
        
        # 获取当前 commit
        current_commit = subprocess.getoutput("git rev-parse --short HEAD")
        print(f"   当前 commit: {current_commit}")
        
        # fetch 并 reset 到最新
        subprocess.run(["git", "fetch", "origin", "main"], 
                      cwd=self.root_dir, check=True, capture_output=True)
        subprocess.run(["git", "reset", "--hard", "origin/main"], 
                      cwd=self.root_dir, check=True, capture_output=True)
        
        # 获取新的 commit
        new_commit = subprocess.getoutput("git rev-parse --short HEAD")
        print(f"✅ 代码已同步: {new_commit}")
        
        if current_commit != new_commit:
            print(f"   更新: {current_commit} → {new_commit}")
        else:
            print("   已是最新版本")
        
        return new_commit
    
    def run_training(self) -> Tuple[bool, str]:
        """
        执行训练
        
        Returns:
            (是否成功, 日志输出)
        """
        print(f"🚀 开始训练 (config: {self.config_file})...")
        
        cmd = [
            "python", "train.py", "train",
            "--path_key", "SRM",
            "--gpu_idx", "0",
            "--net_idx", "TVSRN"
        ]
        
        # 如果指定了非默认配置
        if self.config_file != "config/default.txt":
            config_path = os.path.join("..", self.config_file)
            cmd.extend(["--config", config_path])
        
        print(f"   命令: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd=self.code_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ 训练失败:\n{result.stderr}")
            return False, result.stderr
        
        print("✅ 训练完成")
        return True, result.stdout
    
    def run_validation(self) -> Tuple[bool, Dict[str, float], str]:
        """
        执行验证并提取指标
        
        Returns:
            (是否成功, 指标字典, 日志输出)
        """
        print("📊 开始验证...")
        
        cmd = [
            "python", "val.py", "val",
            "--path_key", "SRM",
            "--gpu_idx", "0",
            "--net_idx", "TVSRN"
        ]
        
        if self.config_file != "config/default.txt":
            config_path = os.path.join("..", self.config_file)
            cmd.extend(["--config", config_path])
        
        result = subprocess.run(
            cmd,
            cwd=self.code_dir,
            capture_output=True,
            text=True
        )
        
        # 解析指标 (根据实际输出格式调整)
        metrics = self._parse_metrics(result.stdout + result.stderr)
        
        if result.returncode != 0:
            print(f"⚠️ 验证可能有问题:\n{result.stderr}")
        
        print(f"✅ 验证完成: PSNR={metrics.get('psnr', 0):.2f}, SSIM={metrics.get('ssim', 0):.4f}")
        return True, metrics, result.stdout
    
    def _parse_metrics(self, output: str) -> Dict[str, float]:
        """
        从训练/验证输出中解析指标
        
        根据实际输出格式，这里使用正则表达式匹配
        """
        metrics = {}
        
        # 匹配 PSNR: xx.xx 或 PSNR=xx.xx
        psnr_match = re.search(r'PSNR[:=]\s*(\d+\.?\d*)', output)
        if psnr_match:
            metrics['psnr'] = float(psnr_match.group(1))
        
        # 匹配 SSIM: xx.xx 或 SSIM=xx.xx
        ssim_match = re.search(r'SSIM[:=]\s*(\d+\.?\d*)', output)
        if ssim_match:
            metrics['ssim'] = float(ssim_match.group(1))
        
        # 匹配 MSE: xx.xx 或 MSE=xx.xx
        mse_match = re.search(r'MSE[:=]\s*(\d+\.?\d*)', output)
        if mse_match:
            metrics['mse'] = float(mse_match.group(1))
        
        # 如果没有匹配到，尝试从文件中读取
        if not metrics:
            metrics = self._load_metrics_from_file()
        
        return metrics
    
    def _load_metrics_from_file(self) -> Dict[str, float]:
        """从模型保存目录加载指标"""
        metrics = {}
        
        # 尝试读取训练日志或指标文件
        model_dir = os.path.join(self.root_dir, "model", "TVSRN")
        if os.path.exists(model_dir):
            # 这里可以根据实际保存的指标文件格式读取
            pass
        
        return metrics
    
    def load_history(self) -> Dict[str, Any]:
        """加载历史指标记录"""
        if os.path.exists(self.metrics_history_file):
            with open(self.metrics_history_file, 'r') as f:
                return json.load(f)
        return {
            "best": {},
            "runs": [],
            "experiments": []
        }
    
    def save_history(self, history: Dict[str, Any]):
        """保存历史指标记录"""
        with open(self.metrics_history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def check_and_archive(self, metrics: Dict[str, float], 
                         git_commit: str, config_name: str) -> Tuple[bool, str]:
        """
        检查是否历史最佳，如果是则存档
        
        Returns:
            (是否改进, 存档路径)
        """
        history = self.load_history()
        best = history.get("best", {})
        
        # 判断标准: PSNR 越高越好
        current_psnr = metrics.get('psnr', 0)
        best_psnr = best.get('psnr', 0)
        improved = current_psnr > best_psnr
        
        archive_path = None
        
        if improved:
            # 创建存档
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_name = f"model_best_{timestamp}"
            archive_path = os.path.join(self.root_dir, "archives", archive_name)
            os.makedirs(archive_path, exist_ok=True)
            
            # 复制模型权重
            model_src = os.path.join(self.root_dir, "model", "TVSRN", "best_model.pth")
            if os.path.exists(model_src):
                subprocess.run(["cp", model_src, os.path.join(archive_path, "model.pth")])
            
            # 保存指标
            archive_metrics = {
                **metrics,
                "git_commit": git_commit,
                "config": config_name,
                "timestamp": datetime.now().isoformat()
            }
            with open(os.path.join(archive_path, "metrics.json"), 'w') as f:
                json.dump(archive_metrics, f, indent=2)
            
            # 保存配置文件副本
            config_src = os.path.join(self.root_dir, self.config_file)
            if os.path.exists(config_src):
                subprocess.run(["cp", config_src, os.path.join(archive_path, "config.txt")])
            
            # 更新历史最佳
            history["best"] = archive_metrics
            print(f"🏆 新历史最佳! PSNR: {best_psnr:.2f} → {current_psnr:.2f}")
            print(f"   已存档: {archive_path}")
        else:
            print(f"📊 当前 PSNR: {current_psnr:.2f}, 最佳: {best_psnr:.2f}")
        
        # 记录本次运行
        run_record = {
            **metrics,
            "git_commit": git_commit,
            "config": config_name,
            "timestamp": datetime.now().isoformat(),
            "improved": improved
        }
        history["runs"].append(run_record)
        self.save_history(history)
        
        return improved, archive_path
    
    def generate_report(self, metrics: Dict[str, float], improved: bool,
                       git_commit: str, config_name: str, archive_path: Optional[str]) -> str:
        """生成训练报告"""
        history = self.load_history()
        best = history.get("best", {})
        
        report_lines = [
            f"# RPLHR-CT 训练报告",
            f"",
            f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**配置**: {config_name}",
            f"**Git Commit**: {git_commit}",
            f"",
            f"## 本次指标",
            f"- PSNR: {metrics.get('psnr', 0):.2f} dB",
            f"- SSIM: {metrics.get('ssim', 0):.4f}",
            f"- MSE: {metrics.get('mse', 0):.6f}",
            f"",
            f"## 历史最佳对比",
            f"- 当前最佳 PSNR: {best.get('psnr', 0):.2f} dB",
            f"- 当前最佳 SSIM: {best.get('ssim', 0):.4f}",
            f"- 最佳模型 Commit: {best.get('git_commit', 'N/A')[:8] if best.get('git_commit') else 'N/A'}",
            f"",
            f"## 是否改进",
        ]
        
        if improved:
            report_lines.append(f"✅ **是** - 已存档为历史最佳模型")
            if archive_path:
                report_lines.append(f"- 存档路径: `{archive_path}`")
        else:
            report_lines.append(f"❌ 否 - 未超越历史最佳")
        
        report_lines.extend([
            f"",
            f"## 累计实验次数",
            f"- 总运行次数: {len(history.get('runs', []))}",
        ])
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"report_{timestamp}.md"
        reports_dir = os.path.join(self.root_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        report_path = os.path.join(reports_dir, report_filename)
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # 同时保存为 latest_report.md 方便下载
        latest_path = os.path.join(reports_dir, "latest_report.md")
        with open(latest_path, 'w') as f:
            f.write(report_content)
        
        print(f"📄 报告已生成: {report_path}")
        return report_path
    
    def run(self) -> Dict[str, Any]:
        """执行完整训练流水线"""
        print("=" * 60)
        print("🚀 RPLHR-CT AutoDL 训练流水线")
        print("=" * 60)
        
        results = {
            "success": False,
            "git_commit": "",
            "metrics": {},
            "improved": False,
            "archive_path": None,
            "report_path": None
        }
        
        try:
            # 1. 同步代码
            git_commit = self.git_sync()
            results["git_commit"] = git_commit
            
            # 2. 训练
            train_success, train_log = self.run_training()
            if not train_success:
                print("❌ 训练失败，终止流程")
                return results
            
            # 3. 验证
            val_success, metrics, val_log = self.run_validation()
            if not val_success:
                print("⚠️ 验证可能有问题，但继续执行")
            
            results["metrics"] = metrics
            
            # 4. 检查并存档
            config_name = os.path.basename(self.config_file)
            improved, archive_path = self.check_and_archive(
                metrics, git_commit, config_name
            )
            results["improved"] = improved
            results["archive_path"] = archive_path
            
            # 5. 生成报告
            report_path = self.generate_report(
                metrics, improved, git_commit, config_name, archive_path
            )
            results["report_path"] = report_path
            results["success"] = True
            
            print("=" * 60)
            print("✅ 流水线执行完成")
            print(f"   PSNR: {metrics.get('psnr', 0):.2f} dB")
            print(f"   是否改进: {'是' if improved else '否'}")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ 流水线执行出错: {e}")
            import traceback
            traceback.print_exc()
        
        return results


if __name__ == "__main__":
    # 从命令行参数获取配置文件
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config/default.txt"
    
    pipeline = TrainingPipeline(config_file)
    results = pipeline.run()
    
    # 输出 JSON 结果供外部调用
    print("\n" + "=" * 60)
    print("RESULTS_JSON:")
    print(json.dumps(results, indent=2))
