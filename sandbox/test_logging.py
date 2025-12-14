#!/usr/bin/env python3
"""
测试sandbox API的日志功能
"""

import requests
import json
import time

API_BASE = "http://127.0.0.1:8000"

def test_sandbox_execution():
    """测试sandbox执行并查看日志"""
    
    # 测试代码
    test_cases = [
        {
            "code": "print('Hello, World!')",
            "language": "python",
            "run_timeout": 30.0
        },
        {
            "code": "print(2 + 2)\nprint('Math is fun!')",
            "language": "python", 
            "run_timeout": 30.0
        },
        {
            "code": "import sys\nprint('Python version:', sys.version)",
            "language": "python",
            "run_timeout": 30.0
        },
        {
            "code": "print('This will cause an error')\nraise ValueError('Test error')",
            "language": "python",
            "run_timeout": 30.0
        }
    ]
    
    print("🧪 开始测试sandbox执行...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 测试用例 {i}:")
        print(f"代码: {test_case['code'][:50]}{'...' if len(test_case['code']) > 50 else ''}")
        
        # 执行代码
        response = requests.post(
            f"{API_BASE}/faas/sandbox/",
            json=test_case,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 执行成功: {result['status']}")
            print(f"输出: {result['run_result']['stdout'][:100]}{'...' if len(result['run_result']['stdout']) > 100 else ''}")
            if result['run_result']['stderr']:
                print(f"错误: {result['run_result']['stderr'][:100]}{'...' if len(result['run_result']['stderr']) > 100 else ''}")
        else:
            print(f"❌ 执行失败: {response.status_code}")
            print(response.text)
        
        time.sleep(0.5)  # 避免执行ID冲突

def test_log_endpoints():
    """测试日志查看端点"""
    
    print("\n📋 测试日志端点...")
    
    # 列出所有日志
    print("\n1. 列出所有日志文件:")
    response = requests.get(f"{API_BASE}/logs/")
    if response.status_code == 200:
        logs_data = response.json()
        print(f"总共找到 {logs_data['total']} 个日志文件")
        
        for log_info in logs_data['logs'][:5]:  # 只显示前5个
            print(f"  - {log_info['filename']} ({log_info['size']} bytes)")
            print(f"    创建时间: {log_info['created']}")
            print(f"    修改时间: {log_info['modified']}")
    else:
        print(f"❌ 获取日志列表失败: {response.status_code}")
        return
    
    # 获取最新的日志详情
    if logs_data['logs']:
        latest_log = logs_data['logs'][0]
        execution_id = latest_log['filename'].replace('execution_', '').replace('.json', '')
        
        print(f"\n2. 获取最新日志详情 (ID: {execution_id}):")
        response = requests.get(f"{API_BASE}/logs/{execution_id}")
        if response.status_code == 200:
            log_detail = response.json()
            print(f"执行ID: {log_detail['execution_id']}")
            print(f"时间戳: {log_detail['timestamp']}")
            print(f"代码:\n{log_detail['code']}")
            print(f"输入: {log_detail['stdin']}")
            print(f"结果状态: {log_detail['result']['status']}")
            print(f"输出: {log_detail['result']['stdout']}")
            if log_detail['result']['stderr']:
                print(f"错误: {log_detail['result']['stderr']}")
        else:
            print(f"❌ 获取日志详情失败: {response.status_code}")

def main():
    """主函数"""
    print("🚀 Sandbox API 日志功能测试")
    print("=" * 50)
    
    try:
        # 测试sandbox执行
        test_sandbox_execution()
        
        # 测试日志端点
        test_log_endpoints()
        
        print("\n✅ 测试完成!")
        print("\n📁 日志文件位置:")
        print("  - 执行日志: ./sandbox_logs/executions.json")
        
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到sandbox API")
        print("请确保API服务正在运行: uvicorn sandbox_api:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")

if __name__ == "__main__":
    main()
