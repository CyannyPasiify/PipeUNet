import argparse
import random
import math
import matplotlib.pyplot as plt
import json
import yaml

# python gen_star_map.py -r 5 -c 8 -sc 0 -ec 5 6 7 -gr 0.2 0.2 -cd 1.5 --step 0.1 -sd 0.3 -s 0.85 0.05 -o star_map.jpg -oc star_map.yaml -rs 0 
def parse_args():
    parser = argparse.ArgumentParser(description='Generate star map')
    parser.add_argument('-sz', '--size', type=int, nargs=2, default=[1366, 768], help='Image size [width, height], default [1366, 768]')
    parser.add_argument('-r', '--row', type=int, required=True, help='Number of rows > 0')
    parser.add_argument('-c', '--col', type=int, required=True, help='Number of columns > 0')
    parser.add_argument('-sr', '--start_row', type=int, nargs='+', default=None, help='Candidate list for start row (0-based, < row)')
    parser.add_argument('-sc', '--start_col', type=int, nargs='+', default=None, help='Candidate list for start column (0-based, < col)')
    parser.add_argument('-er', '--end_row', type=int, nargs='+', default=None, help='Candidate list for end row (0-based, < row)')
    parser.add_argument('-ec', '--end_col', type=int, nargs='+', default=None, help='Candidate list for end column (0-based, < col)')
    parser.add_argument('-rs', '--random_seed', type=int, default=0, help='Random seed, default 0')
    parser.add_argument('-gr', '--gap_ratio', type=float, nargs=2, default=[0.2, 0.2], help='Gap ratio for x and y axes (0-1), default [0.2, 0.2]')
    parser.add_argument('-cd', '--connected_distance', type=float, default=1.0, help='Maximum allowed distance for connected edges, default 1.0')
    parser.add_argument('--step', type=float, default=0.0, help='Step size for increasing maximum allowed distance when not connected, >= 0')
    parser.add_argument('-sd', '--separation_distance', type=float, required=True, help='Minimum separation distance between nodes')
    parser.add_argument('-s', '--sample_probs_per_grid', type=float, nargs='+', required=True, help='Probabilities for number of nodes per grid, sum <= 1')
    parser.add_argument('-o', '--output_map', type=str, default=None, help='Output file path for final result')
    parser.add_argument('-oc', '--output_config', type=str, default=None, help='Output YAML config file path')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.row <= 0:
        parser.error('--row must be > 0')
    if args.col <= 0:
        parser.error('--col must be > 0')
    
    if args.start_row is not None:
        for r in args.start_row:
            if r < 0 or r >= args.row:
                parser.error('--start_row elements must be 0-based and < row')
    
    if args.start_col is not None:
        for c in args.start_col:
            if c < 0 or c >= args.col:
                parser.error('--start_col elements must be 0-based and < col')
    
    if args.end_row is not None:
        for r in args.end_row:
            if r < 0 or r >= args.row:
                parser.error('--end_row elements must be 0-based and < row')
    
    if args.end_col is not None:
        for c in args.end_col:
            if c < 0 or c >= args.col:
                parser.error('--end_col elements must be 0-based and < col')
    
    if args.gap_ratio[0] < 0 or args.gap_ratio[0] > 1 or args.gap_ratio[1] < 0 or args.gap_ratio[1] > 1:
        parser.error('--gap_ratio elements must be between 0 and 1')
    
    if args.step < 0:
        parser.error('--step must be >= 0')
    
    if sum(args.sample_probs_per_grid) > 1:
        parser.error('Sum of --sample_probs_per_grid must be <= 1')
    
    return args


def main():
    args = parse_args()
    random.seed(args.random_seed)
    
    # Print initial parameters
    print(f"Parameters:")
    print(f"  Size: {args.size}")
    print(f"  Rows: {args.row}, Columns: {args.col}")
    print(f"  Start row candidates: {args.start_row}")
    print(f"  Start column candidates: {args.start_col}")
    print(f"  End row candidates: {args.end_row}")
    print(f"  End column candidates: {args.end_col}")
    print(f"  Random seed: {args.random_seed}")
    print(f"  Gap ratio: {args.gap_ratio}")
    print(f"  Connected distance: {args.connected_distance}")
    print(f"  Step: {args.step}")
    print(f"  Separation distance: {args.separation_distance}")
    print(f"  Sample probabilities per grid: {args.sample_probs_per_grid}")
    print(f"  Output map: {args.output_map}")
    print(f"  Output config: {args.output_config}")
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(args.size[0]/100, args.size[1]/100))
    ax.set_xlim(0, args.col)
    ax.set_ylim(0, args.row)
    ax.invert_yaxis()  # Origin at top-left
    
    # Draw grid
    for i in range(args.col + 1):
        ax.axvline(x=i, color='gray', linestyle='--', alpha=0.5)
    for i in range(args.row + 1):
        ax.axhline(y=i, color='gray', linestyle='--', alpha=0.5)
    
    # Sample nodes per grid
    nodes = []
    node_grid_map = {}
    
    print("\nSampling nodes...")
    
    for r in range(args.row):
        for c in range(args.col):
            # Determine number of nodes in this grid
            prob_sum = 0
            num_nodes = 0
            rand_val = random.random()
            
            for i, prob in enumerate(args.sample_probs_per_grid):
                prob_sum += prob
                if rand_val <= prob_sum:
                    num_nodes = i + 1
                    break
            
            if num_nodes > 0:
                grid_nodes = []
                attempts = 0
                max_attempts = 100
                
                while len(grid_nodes) < num_nodes and attempts < max_attempts:
                    # Sample position within grid, considering gap ratio
                    gap_x = args.gap_ratio[0] * 0.5
                    gap_y = args.gap_ratio[1] * 0.5
                    x = c + gap_x + random.random() * (1 - 2 * gap_x)
                    y = r + gap_y + random.random() * (1 - 2 * gap_y)
                    
                    # Check separation distance with existing nodes in the same grid
                    valid = True
                    for (nx, ny) in grid_nodes:
                        distance = math.sqrt((x - nx)**2 + (y - ny)**2)
                        if distance < args.separation_distance:
                            valid = False
                            break
                    
                    if valid:
                        grid_nodes.append((x, y))
                    attempts += 1
                
                nodes.extend(grid_nodes)
                node_grid_map[(r, c)] = grid_nodes
                print(f"  Grid ({r}, {c}): {len(grid_nodes)} nodes sampled")
    
    # Global separation distance check
    print("\nChecking global separation distances...")
    nodes_changed = True
    while nodes_changed:
        nodes_changed = False
        to_remove = []
        
        for i in range(len(nodes)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(nodes)):
                if j in to_remove:
                    continue
                distance = math.sqrt((nodes[i][0] - nodes[j][0])**2 + (nodes[i][1] - nodes[j][1])**2)
                if distance < args.separation_distance:
                    # Randomly remove one node
                    if random.random() < 0.5:
                        to_remove.append(i)
                    else:
                        to_remove.append(j)
                    nodes_changed = True
                    break
        
        # Remove nodes in reverse order to avoid index issues
        for idx in sorted(to_remove, reverse=True):
            nodes.pop(idx)
        
        if nodes_changed:
            print(f"  Removed {len(to_remove)} nodes due to separation distance violation")
    
    # Select start and end nodes
    print("\nSelecting start and end nodes...")
    
    # Filter nodes by start row and column candidates
    start_candidates = []
    for node in nodes:
        r = int(node[1])
        c = int(node[0])
        if (args.start_row is None or r in args.start_row) and (args.start_col is None or c in args.start_col):
            start_candidates.append(node)
    
    if not start_candidates:
        print("  No start candidates found")
        return
    
    start_node = random.choice(start_candidates)
    print(f"  Start node: {start_node}")
    
    # Filter nodes by end row and column candidates, excluding start node
    end_candidates = []
    for node in nodes:
        if node == start_node:
            continue
        r = int(node[1])
        c = int(node[0])
        if (args.end_row is None or r in args.end_row) and (args.end_col is None or c in args.end_col):
            end_candidates.append(node)
    
    if not end_candidates:
        print("  No end candidates found")
        return
    
    end_node = random.choice(end_candidates)
    print(f"  End node: {end_node}")
    
    # Check connectivity
    print("\nChecking connectivity...")
    current_distance = args.connected_distance
    connected = False
    
    while not connected:
        # Build adjacency list
        adjacency = {i: [] for i in range(len(nodes))}
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                distance = math.sqrt((nodes[i][0] - nodes[j][0])**2 + (nodes[i][1] - nodes[j][1])**2)
                if distance <= current_distance:
                    adjacency[i].append(j)
                    adjacency[j].append(i)
        
        # Check if start and end are connected using BFS
        start_idx = nodes.index(start_node)
        end_idx = nodes.index(end_node)
        
        visited = set()
        queue = [start_idx]
        visited.add(start_idx)
        
        while queue:
            current = queue.pop(0)
            if current == end_idx:
                connected = True
                break
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        if connected:
            print(f"  Connected with distance: {current_distance}")
        else:
            print(f"  Not connected with distance: {current_distance}")
            if args.step > 0:
                current_distance += args.step
                print(f"  Increasing distance to: {current_distance}")
            else:
                print("  Cannot connect with given parameters, exiting")
                return
    
    # Remove isolated nodes (nodes not connected to start)
    print("\nRemoving isolated nodes...")
    connected_indices = list(visited)
    print(f"  Found {len(connected_indices)} nodes connected to start")
    print(f"  Removing {len(nodes) - len(connected_indices)} isolated nodes")
    
    # Create mapping from old indices to new indices
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(connected_indices)}
    
    # Rebuild nodes with new indices
    new_nodes = [nodes[i] for i in connected_indices]
    new_start_idx = old_to_new[start_idx]
    new_end_idx = old_to_new[end_idx]
    
    # Rebuild edges with new indices
    new_edges = []
    for i in range(len(new_nodes)):
        for j in range(i + 1, len(new_nodes)):
            old_i = connected_indices[i]
            old_j = connected_indices[j]
            distance = math.sqrt((nodes[old_i][0] - nodes[old_j][0])**2 + (nodes[old_i][1] - nodes[old_j][1])**2)
            if distance <= current_distance:
                new_edges.append((i, j))
    
    print(f"  Created {len(new_edges)} edges")
    
    # Rebuild nodes as dictionary with sequence numbers
    print("\nBuilding node dictionary...")
    nodes_dict = {}
    for seq, (x, y) in enumerate(new_nodes):
        grid_x = int(x)
        grid_y = int(y)
        nodes_dict[seq] = [x, y, grid_x, grid_y]
    
    print(f"  Node dictionary created with {len(nodes_dict)} nodes")
    print(f"  Entry (start) node: {new_start_idx}")
    print(f"  Exit (end) node: {new_end_idx}")
    
    # Draw nodes
    ax.scatter([n[0] for n in new_nodes], [n[1] for n in new_nodes], c='blue', s=20)
    ax.scatter(new_nodes[new_start_idx][0], new_nodes[new_start_idx][1], c='green', s=50, marker='s')
    ax.scatter(new_nodes[new_end_idx][0], new_nodes[new_end_idx][1], c='red', s=50, marker='s')
    
    # Draw edges
    for edge in new_edges:
        node1 = new_nodes[edge[0]]
        node2 = new_nodes[edge[1]]
        ax.plot([node1[0], node2[0]], [node1[1], node2[1]], 'gray', alpha=0.5)
    
    # Save or display result
    if args.output_map or args.output_config:
        # Prepare config data
        config_data = {
            'entry': new_start_idx,
            'exit': new_end_idx,
            'nodes': nodes_dict,
            'edges': new_edges,
            'parameters': {
                'size': args.size,
                'row': args.row,
                'col': args.col,
                'start_row': args.start_row,
                'start_col': args.start_col,
                'end_row': args.end_row,
                'end_col': args.end_col,
                'random_seed': args.random_seed,
                'gap_ratio': args.gap_ratio,
                'connected_distance': current_distance,
                'step': args.step,
                'separation_distance': args.separation_distance,
                'sample_probs_per_grid': args.sample_probs_per_grid
            }
        }
    
    if args.output_map:
        print(f"\nSaving result to: {args.output_map}")
        # Save as image
        plt.savefig(args.output_map, dpi=100, bbox_inches='tight')
    
    if args.output_config:
        print(f"\nSaving config to: {args.output_config}")
        # Convert data to YAML-friendly format
        yaml_config = {
            'entry': new_start_idx,
            'exit': new_end_idx,
            'nodes': nodes_dict,
            'edges': [list(edge) for edge in new_edges],  # Convert tuples to lists
            'parameters': {
                'size': args.size,
                'row': args.row,
                'col': args.col,
                'start_row': args.start_row,
                'start_col': args.start_col,
                'end_row': args.end_row,
                'end_col': args.end_col,
                'random_seed': args.random_seed,
                'gap_ratio': args.gap_ratio,
                'connected_distance': current_distance,
                'step': args.step,
                'separation_distance': args.separation_distance,
                'sample_probs_per_grid': args.sample_probs_per_grid
            }
        }
        # Save with Pythonic YAML style
        with open(args.output_config, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=None, sort_keys=False, allow_unicode=True)
        print(f"  Config saved successfully")
    
    # Show plot
    plt.title('Star Map')
    plt.show()


if __name__ == '__main__':
    main()
