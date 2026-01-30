#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 -o/--model_dir指定K-Means模型参数保存目录，对于每个k值包含一个{k}_means_anchors.yaml文件，其中anchors属性以列表形式记录了K-Means模型每个锚特征向量，当前K值下的SSE、Silhouette、CH、DB值记录为属性[sse,silhouette,ch,db]。 
 遍历每个k值对应的文件，并遍历-s/--score指定的筛选指标，默认为[sse,silhouette,ch,db]，为每个指标绘制一张横坐标为K值纵坐标为score名称的折线图，输出到-o/--output_graph_dir目录，命名格式为graph_k_{score}.jpg。
"""

"""
Plot K-Means Model Score Graphs

This script plots evaluation metrics for K-Means clustering models.

Parameters:
    -o, --model_dir: Directory containing K-Means model parameter YAML files
    -s, --score: List of evaluation metrics to plot (default: ['sse', 'silhouette', 'ch', 'db'])
    -o, --output_graph_dir: Directory to save generated graphs

Usage Examples:
    python 05_plot_model_score_graph.py --model_dir /path/to/models --output_graph_dir /path/to/graphs
    python 05_plot_model_score_graph.py --model_dir /path/to/models --score silhouette ch --output_graph_dir /path/to/graphs
"""

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Plot evaluation metrics for K-Means clustering models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model_dir /path/to/models --output_graph_dir /path/to/graphs
  %(prog)s --model_dir /path/to/models --score silhouette ch --output_graph_dir /path/to/graphs
        """
    )
    
    parser.add_argument(
        '-m', '--model_dir',
        type=str,
        required=True,
        help='Directory containing K-Means model parameter YAML files'
    )
    
    parser.add_argument(
        '-s', '--score',
        type=str,
        nargs='+',
        default=['sse', 'silhouette', 'ch', 'db'],
        help='List of evaluation metrics to plot (default: [\'sse\', \'silhouette\', \'ch\', \'db\'])'
    )
    
    parser.add_argument(
        '-o', '--output_graph_dir',
        type=str,
        required=True,
        help='Directory to save generated graphs'
    )
    
    return parser.parse_args()


def read_kmeans_results(model_dir: str) -> Dict[int, Dict[str, float]]:
    """
    Read K-Means results from YAML files.
    
    Args:
        model_dir: Directory containing K-Means model parameter YAML files
        
    Returns:
        Dict[int, Dict[str, float]]: Dictionary with k values as keys and evaluation metrics as values
    """
    model_path: Path = Path(model_dir)
    results: Dict[int, Dict[str, float]] = {}
    
    # Find all YAML files matching the pattern
    yaml_files: List[Path] = list(model_path.glob("*_means_anchors.yaml"))
    
    for file_path in yaml_files:
        # Extract k value from filename
        filename: str = file_path.stem
        k_str: str = filename.split("_")[0]
        try:
            k: int = int(k_str)
        except ValueError:
            print(f"Skipping file with invalid k value: {file_path}")
            continue
        
        # Read YAML content
        try:
            with open(file_path, 'r') as f:
                data: Dict = yaml.safe_load(f)
            
            # Extract evaluation metrics
            metrics: Dict[str, float] = {}
            for metric in ['sse', 'silhouette', 'ch', 'db']:
                if metric in data:
                    metrics[metric] = data[metric]
            
            results[k] = metrics
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue
    
    return results


def plot_metrics(results: Dict[int, Dict[str, float]], metrics: List[str], output_dir: str) -> None:
    """
    Plot and save graphs for each specified metric.
    
    Args:
        results: Dictionary with k values as keys and evaluation metrics as values
        metrics: List of evaluation metrics to plot
        output_dir: Directory to save generated graphs
    """
    output_path: Path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sort k values for plotting
    sorted_k: List[int] = sorted(results.keys())
    
    if not sorted_k:
        print("No valid k values found in results.")
        return
    
    for metric in metrics:
        # Check if metric exists in any result
        has_data: bool = any(metric in results[k] for k in sorted_k)
        if not has_data:
            print(f"No data found for metric: {metric}")
            continue
        
        # Extract metric values
        metric_values: List[float] = []
        valid_k: List[int] = []
        
        for k in sorted_k:
            if metric in results[k]:
                metric_values.append(results[k][metric])
                valid_k.append(k)
        
        if not valid_k:
            print(f"No valid data points for metric: {metric}")
            continue
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(valid_k, metric_values, marker='o', linestyle='-', linewidth=2, markersize=8)
        
        # Set plot properties
        plt.xlabel('Number of Clusters (k)', fontsize=14)
        plt.ylabel(metric.upper(), fontsize=14)
        plt.title(f'K-Means {metric.upper()} vs Number of Clusters', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Customize x-axis ticks
        plt.xticks(valid_k)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_file: Path = output_path / f"graph_k_{metric}.jpg"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved graph for {metric} to: {output_file}")


def main() -> None:
    """
    Main function to orchestrate the graph plotting workflow.
    """
    # Parse command line arguments
    args: argparse.Namespace = parse_args()
    
    print(f"Model directory: {args.model_dir}")
    print(f"Metrics to plot: {args.score}")
    print(f"Output graph directory: {args.output_graph_dir}")
    
    # Read K-Means results from YAML files
    print("\nReading K-Means results...")
    results: Dict[int, Dict[str, float]] = read_kmeans_results(args.model_dir)
    
    if not results:
        print("No valid results found.")
        return
    
    # Plot and save metrics
    print("\nPlotting metrics...")
    plot_metrics(results, args.score, args.output_graph_dir)
    
    print("\nGraph plotting completed successfully!")


if __name__ == '__main__':
    main()