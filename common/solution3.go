package common

// 83. 删除排序链表中的重复元素 保留一个
func deleteDuplicates(head *ListNode) *ListNode {
	var dfs func(node *ListNode) *ListNode
	dfs = func(node *ListNode) *ListNode {
		if node == nil || node.Next == nil {
			return node
		}
		if node.Val != node.Next.Val {
			node.Next = dfs(node.Next)
			return node
		} else {
			val := node.Val
			for node != nil && node.Val == val {
				node = node.Next
			}
			return &ListNode{Val: val, Next: dfs(node)}
		}
	}
	return dfs(head)
}

// 110. 平衡二叉树 左右子树高度差不超过1
func isBalanced(root *TreeNode) bool {
	var dfs func(node *TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		var left, right int
		if left = dfs(node.Left); left == -1 {
			return left
		}
		if right = dfs(node.Right); right == -1 {
			return right
		}
		if Abs(left-right) > 1 {
			return -1
		}
		return max(left, right) + 1
	}
	return dfs(root) != -1
}

// 404. 左叶子之和
func sumOfLeftLeaves(root *TreeNode) int {
	var sum int
	var dfs func(node *TreeNode, flag bool)
	dfs = func(node *TreeNode, flag bool) {
		if node == nil {
			return
		}
		if flag && node.Left == nil && node.Right == nil {
			sum += node.Val
		}
		dfs(node.Left, true)
		dfs(node.Right, false)
	}
	dfs(root, false)
	return sum
}

// 416. 分割等和子集 01背包问题 先遍历物品 再倒序遍历背包
func canPartition(nums []int) bool {
	var sum int
	for _, num := range nums {
		sum += num
	}
	if sum%2 != 0 {
		return false
	}
	target := sum / 2
	//dpi 容量为i的背包所能装的最大价值
	dp := make([]int, target+1)
	for i := 0; i < len(nums); i++ {
		for j := target; j >= nums[i]; j-- {
			dp[j] = max(dp[j], dp[j-nums[i]]+nums[i])
		}
	}
	return dp[target] == target
}
