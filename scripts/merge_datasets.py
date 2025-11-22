#!/usr/bin/env python3
"""
合并 openr1 和 guru_rl92k 数据集，并去除重复题目
基于 user content 进行去重
支持合并训练集和测试集
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List, Optional


def extract_user_content(prompt):
    """从prompt数组中提取user角色的content"""
    if isinstance(prompt, (list, np.ndarray)):
        for item in prompt:
            if isinstance(item, dict) and item.get('role') == 'user':
                return item.get('content', '')
    return ''


def merge_test_datasets(openr1_test_path: str, guru_test_paths: List[str], 
                        output_path: str, keep: str = 'first'):
    """
    合并测试集（验证集）并去重
    
    Args:
        openr1_test_path: openr1测试集路径
        guru_test_paths: guru_rl92k测试集路径列表（可以有多个文件）
        output_path: 输出路径
        keep: 去重时保留的策略
    """
    print("=" * 60)
    print("测试集合并和去重工具")
    print("=" * 60)
    
    # 读取openr1测试集
    print(f"\n[1/5] 正在读取测试集...")
    df_openr1 = pd.read_parquet(openr1_test_path)
    print(f"  - openr1 test: {df_openr1.shape[0]} 条记录, {df_openr1.shape[1]} 列")
    
    # 读取所有guru测试集
    df_guru_list = []
    for i, path in enumerate(guru_test_paths):
        df = pd.read_parquet(path)
        df_guru_list.append(df)
        print(f"  - guru test[{i+1}]: {df.shape[0]} 条记录, {df.shape[1]} 列")
    
    # 合并所有guru测试集
    if len(df_guru_list) > 1:
        df_guru = pd.concat(df_guru_list, ignore_index=True)
    else:
        df_guru = df_guru_list[0]
    
    print(f"  - guru_rl92k 总计: {df_guru.shape[0]} 条记录")
    
    # 提取user content用于去重
    print(f"\n[2/5] 正在提取user content用于去重...")
    df_openr1['user_content'] = df_openr1['prompt'].apply(extract_user_content)
    df_guru['user_content'] = df_guru['prompt'].apply(extract_user_content)
    
    # 添加数据源标识
    df_openr1['_source_dataset'] = 'openr1'
    df_guru['_source_dataset'] = 'guru_rl92k'
    
    # 统一列结构
    print(f"\n[3/5] 正在统一列结构...")
    
    # 获取所有可能的列
    all_cols_set = set(df_openr1.columns) | set(df_guru.columns)
    all_cols_set.discard('user_content')
    all_cols_set.discard('_source_dataset')
    
    # 为openr1添加缺失的列
    for col in all_cols_set:
        if col not in df_openr1.columns:
            df_openr1[col] = None
    
    # 为guru添加缺失的列
    for col in all_cols_set:
        if col not in df_guru.columns:
            df_guru[col] = None
    
    # 统一列顺序
    priority_cols = sorted([c for c in all_cols_set])
    final_cols = priority_cols + ['user_content', '_source_dataset']
    
    df_openr1 = df_openr1[final_cols]
    df_guru = df_guru[final_cols]
    
    # 合并数据集
    print(f"\n[4/5] 正在合并数据集...")
    
    if keep == 'guru':
        df_merged = df_guru.copy()
        openr1_unique = df_openr1[~df_openr1['user_content'].isin(df_guru['user_content'])]
        df_merged = pd.concat([df_merged, openr1_unique], ignore_index=True)
        print(f"  策略: 优先保留guru数据，然后添加openr1中不重复的数据")
    elif keep == 'openr1':
        df_merged = df_openr1.copy()
        guru_unique = df_guru[~df_guru['user_content'].isin(df_openr1['user_content'])]
        df_merged = pd.concat([df_merged, guru_unique], ignore_index=True)
        print(f"  策略: 优先保留openr1数据，然后添加guru中不重复的数据")
    else:
        df_merged = pd.concat([df_openr1, df_guru], ignore_index=True)
        df_merged = df_merged.drop_duplicates(subset=['user_content'], keep=keep)
        print(f"  策略: 使用pandas drop_duplicates，keep='{keep}'")
    
    print(f"  - 合并后: {df_merged.shape[0]} 条记录, {df_merged.shape[1]} 列")
    
    # 统计信息
    print(f"\n[5/5] 统计信息:")
    print(f"  - 原始总数: {df_openr1.shape[0] + df_guru.shape[0]} 条")
    print(f"  - 去重后: {df_merged.shape[0]} 条")
    print(f"  - 去除重复: {df_openr1.shape[0] + df_guru.shape[0] - df_merged.shape[0]} 条")
    
    source_counts = df_merged['_source_dataset'].value_counts()
    print(f"  - 数据来源分布:")
    for source, count in source_counts.items():
        print(f"    {source}: {count} 条 ({count/df_merged.shape[0]*100:.2f}%)")
    
    # 保存结果
    print(f"\n正在保存到: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_merged_final = df_merged.drop(columns=['user_content'])
    df_merged_final.to_parquet(output_path, index=False)
    print(f"✓ 保存完成!")
    
    return df_merged_final


def merge_and_deduplicate(openr1_path, guru_path, output_path, keep='first'):
    """
    合并两个数据集并去重
    
    Args:
        openr1_path: openr1数据集路径
        guru_path: guru_rl92k数据集路径
        output_path: 输出路径
        keep: 去重时保留的策略 ('first', 'last', 'guru', 'openr1')
              'guru'表示优先保留guru数据，'openr1'表示优先保留openr1数据
    """
    print("=" * 60)
    print("数据集合并和去重工具")
    print("=" * 60)
    
    # 读取数据集
    print(f"\n[1/5] 正在读取数据集...")
    df_openr1 = pd.read_parquet(openr1_path)
    df_guru = pd.read_parquet(guru_path)
    
    print(f"  - openr1: {df_openr1.shape[0]} 条记录, {df_openr1.shape[1]} 列")
    print(f"  - guru_rl92k: {df_guru.shape[0]} 条记录, {df_guru.shape[1]} 列")
    
    # 提取user content用于去重
    print(f"\n[2/5] 正在提取user content用于去重...")
    df_openr1['user_content'] = df_openr1['prompt'].apply(extract_user_content)
    df_guru['user_content'] = df_guru['prompt'].apply(extract_user_content)
    
    # 添加数据源标识
    df_openr1['_source_dataset'] = 'openr1'
    df_guru['_source_dataset'] = 'guru_rl92k'
    
    # 统一列结构
    print(f"\n[3/5] 正在统一列结构...")
    
    # 获取所有列
    common_cols = ['prompt', 'data_source', 'ability', 'reward_model', 'extra_info', 
                   'user_content', '_source_dataset']
    openr1_only_cols = []
    guru_only_cols = ['source', 'domain', 'llama8b_solve_rate', 'apply_chat_template', 
                      'is_unique', 'solution', 'qwen2.5_7b_pass_rate', 'qwen3_30b_pass_rate']
    
    # 为openr1添加缺失的列（填充None）
    for col in guru_only_cols:
        if col not in df_openr1.columns:
            df_openr1[col] = None
    
    # 为guru添加缺失的列（如果有的话）
    for col in openr1_only_cols:
        if col not in df_guru.columns:
            df_guru[col] = None
    
    # 选择要保留的列（按guru的列顺序，但包含所有列）
    all_cols = list(df_guru.columns) + [col for col in df_openr1.columns if col not in df_guru.columns]
    # 确保user_content和_source_dataset在最后
    priority_cols = [c for c in all_cols if c not in ['user_content', '_source_dataset']]
    final_cols = priority_cols + ['user_content', '_source_dataset']
    
    # 重新排列列
    df_openr1 = df_openr1[[c for c in final_cols if c in df_openr1.columns]]
    df_guru = df_guru[[c for c in final_cols if c in df_guru.columns]]
    
    # 合并数据集
    print(f"\n[4/5] 正在合并数据集...")
    
    if keep == 'guru':
        # 优先保留guru数据：先添加guru，再添加openr1中不重复的
        df_merged = df_guru.copy()
        openr1_unique = df_openr1[~df_openr1['user_content'].isin(df_guru['user_content'])]
        df_merged = pd.concat([df_merged, openr1_unique], ignore_index=True)
        print(f"  策略: 优先保留guru数据，然后添加openr1中不重复的数据")
    elif keep == 'openr1':
        # 优先保留openr1数据：先添加openr1，再添加guru中不重复的
        df_merged = df_openr1.copy()
        guru_unique = df_guru[~df_guru['user_content'].isin(df_openr1['user_content'])]
        df_merged = pd.concat([df_merged, guru_unique], ignore_index=True)
        print(f"  策略: 优先保留openr1数据，然后添加guru中不重复的数据")
    else:
        # 使用pandas的去重方法
        df_merged = pd.concat([df_openr1, df_guru], ignore_index=True)
        # 基于user_content去重
        df_merged = df_merged.drop_duplicates(subset=['user_content'], keep=keep)
        print(f"  策略: 使用pandas drop_duplicates，keep='{keep}'")
    
    print(f"  - 合并后: {df_merged.shape[0]} 条记录, {df_merged.shape[1]} 列")
    
    # 统计信息
    print(f"\n[5/5] 统计信息:")
    print(f"  - 原始总数: {df_openr1.shape[0] + df_guru.shape[0]} 条")
    print(f"  - 去重后: {df_merged.shape[0]} 条")
    print(f"  - 去除重复: {df_openr1.shape[0] + df_guru.shape[0] - df_merged.shape[0]} 条")
    
    source_counts = df_merged['_source_dataset'].value_counts()
    print(f"  - 数据来源分布:")
    for source, count in source_counts.items():
        print(f"    {source}: {count} 条 ({count/df_merged.shape[0]*100:.2f}%)")
    
    # 保存结果
    print(f"\n正在保存到: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 移除临时列（可选，这里保留_source_dataset用于追踪）
    df_merged_final = df_merged.drop(columns=['user_content'])  # 移除用于去重的临时列
    
    df_merged_final.to_parquet(output_path, index=False)
    print(f"✓ 保存完成!")
    
    return df_merged_final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='合并openr1和guru_rl92k数据集并去重')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'],
                       default='both',
                       help='合并模式: train(仅训练集), test(仅测试集), both(训练集和测试集)')
    parser.add_argument('--openr1-train', type=str, 
                       default='/home/chenluy/data/openr1/train.parquet',
                       help='openr1训练集路径')
    parser.add_argument('--guru-train', type=str,
                       default='/home/chenluy/data/guru_rl92k/train/math__combined_54.4k.parquet',
                       help='guru_rl92k训练集路径')
    parser.add_argument('--openr1-test', type=str,
                       default='/home/chenluy/data/openr1/test.parquet',
                       help='openr1测试集路径')
    parser.add_argument('--guru-test', type=str, nargs='+',
                       default=['/home/chenluy/data/guru_rl92k/online_eval/math__math_500.parquet',
                               '/home/chenluy/data/guru_rl92k/online_eval/math__aime_repeated_8x_240.parquet'],
                       help='guru_rl92k测试集路径（可指定多个文件）')
    parser.add_argument('--output-train', type=str,
                       default='/home/chenluy/data/merged_openr1_guru/train.parquet',
                       help='训练集输出路径')
    parser.add_argument('--output-test', type=str,
                       default='/home/chenluy/data/merged_openr1_guru/test.parquet',
                       help='测试集输出路径')
    parser.add_argument('--keep', type=str, choices=['first', 'last', 'guru', 'openr1'],
                       default='guru',
                       help='去重策略: first/last使用pandas默认, guru/openr1优先保留指定数据集')
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        print("\n" + "="*60)
        print("合并训练集")
        print("="*60)
        merge_and_deduplicate(args.openr1_train, args.guru_train, 
                            args.output_train, args.keep)
    
    if args.mode in ['test', 'both']:
        print("\n" + "="*60)
        print("合并测试集")
        print("="*60)
        merge_test_datasets(args.openr1_test, args.guru_test,
                          args.output_test, args.keep)

