package main

import (
	"math"
)

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
func getLeft(node *TreeNode) int {
	for node.Left != nil {
		node = node.Left
	}
	return node.Val
}
func getRight(node *TreeNode) int {
	for node.Right != nil {
		node = node.Right
	}
	return node.Val
}

// 验证二叉搜索树
//func isValidBST(root *TreeNode) bool {
//	//一个节点需要小于右子树的任意一个节点
//	//也就是说需要小于右边的最小的
//	//同理 需要大于左边的最大的
//	var dfs func(node *TreeNode) bool
//	dfs = func(node *TreeNode) bool {
//		if node == nil {
//			return true
//		}
//		//如果左子树和右子树都是合法的二叉搜索树 再去下一步判断
//		//否则直接返回false
//		if dfs(node.Left) && dfs(node.Right) {
//			//当前节点需要小于右边的最小的 大于左边的最大的
//			return (node.Right == nil || node.Val < getLeft(node.Right)) && (node.Left == nil || node.Val > getRight(node.Left))
//		} else {
//			return false
//		}
//	}
//	return dfs(root)
//}

// 二叉搜索树的中序遍历为升序
func isValidBST(root *TreeNode) bool {
	sequences := make([]int, 0, 1000)
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		sequences = append(sequences, node.Val)
		dfs(node.Right)
	}
	dfs(root)
	//判断sequences是否严格递增
	for i := 1; i < len(sequences); i++ {
		if !(sequences[i] > sequences[i-1]) {
			return false
		}
	}
	return true
}

// 二叉搜索树的最小绝对差
func getMinimumDifference(root *TreeNode) int {
	//中序遍历 时刻维护上一个节点的值
	ans := math.MaxInt
	prev := -1
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		if prev != -1 {
			ans = min(ans, Abs(node.Val-prev))
		}
		prev = node.Val
		dfs(node.Right)
	}
	dfs(root)
	return ans
}

// 二叉搜索树中的众数
func findMode(root *TreeNode) (ans []int) {
	prev := -1
	timesCnt := 1
	maxTimes := 1
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		if node.Val == prev {
			timesCnt++
		} else {
			timesCnt = 1
		}
		if timesCnt == maxTimes {
			ans = append(ans, node.Val)
		} else if timesCnt > maxTimes {
			ans = []int{node.Val}
			maxTimes = timesCnt
		}
		prev = node.Val
		dfs(node.Right)
	}
	dfs(root)
	return
}

// 二叉树的最近公共祖先
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	var dfs func(node *TreeNode) *TreeNode
	dfs = func(node *TreeNode) *TreeNode {
		if node == nil || node == p || node == q {
			return node
		}
		//如果递归左子树的结果不为空 说明p或者q在左子树
		left := dfs(node.Left)
		//如果递归右子树的结果不为空 说明p或者q在右子树
		right := dfs(node.Right)
		//两个都不为空 说明一个在左一个在右 直接返回当前节点就是lca
		if left != nil && right != nil {
			return node
		}
		//如果左子树为空 两个节点都在右子树
		if left == nil {
			return right
		}
		//两个节点都在左子树
		return left
	}
	return dfs(root)
}
