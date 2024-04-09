package main

import "math"

func getMaximum(nums []int) (int, int) {
	maximum := math.MinInt8
	index := -1
	for i, num := range nums {
		if num > maximum {
			index = i
			maximum = num
		}
	}
	return index, maximum
}

// 最大二叉树
func constructMaximumBinaryTree(nums []int) *TreeNode {
	if len(nums) == 0 {
		return nil
	}
	//找到最大值的index
	index, maximum := getMaximum(nums)
	return &TreeNode{
		Val:   maximum,
		Left:  constructMaximumBinaryTree(nums[:index]),
		Right: constructMaximumBinaryTree(nums[index+1:]),
	}
}

// 合并二叉树
func mergeTrees(root1 *TreeNode, root2 *TreeNode) *TreeNode {
	var dfs func(node1 *TreeNode, node2 *TreeNode) *TreeNode
	dfs = func(node1 *TreeNode, node2 *TreeNode) *TreeNode {
		if node1 == nil || node2 == nil {
			if node1 == nil {
				return node2
			}
			return node1
		}
		node1.Val += node2.Val
		node1.Left = dfs(node1.Left, node2.Left)
		node1.Right = dfs(node1.Right, node2.Right)
		return node1
	}
	return dfs(root1, root2)
}

// 二叉搜索树中的搜索
func searchBST(root *TreeNode, val int) *TreeNode {
	var dfs func(node *TreeNode) *TreeNode
	dfs = func(node *TreeNode) *TreeNode {
		if node == nil {
			return nil
		}
		if node.Val < val {
			return dfs(node.Right)
		} else if node.Val > val {
			return dfs(node.Left)
		} else {
			return node
		}
	}
	return dfs(root)
}
