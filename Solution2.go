package main

import (
	"container/heap"
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

// 如果二叉树是二叉搜索树
func lowestCommonAncestor2(root, p, q *TreeNode) *TreeNode {
	//二叉搜索树可以通过值来判断 当前节点的值如果比pq都小 递归右子树
	//如果都大 递归左子树
	//处于二者之间 直接返回
	var dfs func(node *TreeNode) *TreeNode
	dfs = func(node *TreeNode) *TreeNode {
		if node == nil || node == p || node == q {
			return node
		}
		if node.Val < p.Val && node.Val < q.Val {
			return dfs(node.Right)
		} else if node.Val > p.Val && node.Val > q.Val {
			return dfs(node.Left)
		} else {
			return node
		}
	}
	return dfs(root)
}

// 二叉搜索树的插入操作
//
//	func insertIntoBST(root *TreeNode, val int) *TreeNode {
//		var dfs func(node *TreeNode)
//		dfs = func(node *TreeNode) {
//			if val > node.Val {
//				if node.Right == nil {
//					node.Right = &TreeNode{Val: val}
//				} else {
//					dfs(node.Right)
//				}
//			} else {
//				if node.Left == nil {
//					node.Left = &TreeNode{Val: val}
//				} else {
//					dfs(node.Left)
//				}
//			}
//		}
//		if root == nil {
//			return &TreeNode{Val: val}
//		}
//		dfs(root)
//		return root
//	}
//
// 更简单的方法
func insertIntoBST(root *TreeNode, val int) *TreeNode {
	if root == nil {
		return &TreeNode{Val: val}
	}
	if root.Val > val {
		root.Left = insertIntoBST(root.Left, val)
	} else {
		root.Right = insertIntoBST(root.Right, val)
	}
	return root
}

// 删除二叉搜索树中的节点
func deleteNode(root *TreeNode, key int) *TreeNode {
	if root == nil {
		return root
	}
	if root.Val == key {
		if root.Left != nil {
			temp := root.Left
			for temp.Right != nil {
				temp = temp.Right
			}
			temp.Right = root.Right
			return root.Left
		}
		return root.Right
	}
	if root.Val < key {
		root.Right = deleteNode(root.Right, key)
	} else if root.Val > key {
		root.Left = deleteNode(root.Left, key)
	}
	return root
}

// 修建二叉搜索树
func trimBST(root *TreeNode, low int, high int) *TreeNode {
	//如果当前节点小于low 其左子树更小 也需要删掉 直接返回其右子树的修剪结果
	//如果当前节点大于high 其右子树更大 也需要删掉
	//如果当前节点处于两者之间 当前节点不需要删除 但是需要分别递归删除其左右子树
	if root == nil {
		return nil
	}
	if root.Val < low {
		return trimBST(root.Right, low, high)
	} else if root.Val > high {
		return trimBST(root.Left, low, high)
	} else {
		root.Left = trimBST(root.Left, low, high)
		root.Right = trimBST(root.Right, low, high)
		return root
	}
}

// 将有序数组转换为二叉搜索树
func sortedArrayToBST(nums []int) *TreeNode {
	if len(nums) == 0 {
		return nil
	}
	mid := len(nums) / 2
	return &TreeNode{
		Val:   nums[mid],
		Left:  sortedArrayToBST(nums[:mid]),
		Right: sortedArrayToBST(nums[mid+1:]),
	}
}

// 二叉搜索树转换为累加树
func convertBST(root *TreeNode) *TreeNode {
	sum := 0
	//中序遍历 倒着返回
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Right)
		sum += node.Val
		node.Val = sum
		dfs(node.Left)
	}
	dfs(root)
	return root
}

type NodeHeap []*ListNode

func (n *NodeHeap) Len() int {
	return len(*n)
}

func (n *NodeHeap) Less(i, j int) bool {
	return (*n)[i].Val < (*n)[j].Val
}

func (n *NodeHeap) Swap(i, j int) {
	(*n)[i], (*n)[j] = (*n)[j], (*n)[i]
}

func (n *NodeHeap) Push(x any) {
	*n = append(*n, x.(*ListNode))
}

func (n *NodeHeap) Pop() any {
	x := (*n)[n.Len()-1]
	*n = (*n)[:n.Len()-1]
	return x
}

// 堆排序解法
func mergeKLists(lists []*ListNode) *ListNode {
	dummy := &ListNode{}
	cur := dummy
	hp := &NodeHeap{}
	for _, list := range lists {
		if list != nil {
			heap.Push(hp, list)
		}
	}
	for hp.Len() > 0 {
		popNode := heap.Pop(hp).(*ListNode)
		cur.Next = &ListNode{Val: popNode.Val}
		if popNode.Next != nil {
			heap.Push(hp, popNode.Next)
		}
		cur = cur.Next
	}
	return dummy.Next
}

// 电话号码的字母组合
func letterCombinations(digits string) (ans []string) {
	mapping := []string{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"}
	var dfs func(i int, path []byte)
	dfs = func(i int, path []byte) {
		if i == len(digits) {
			ans = append(ans, string(path))
			return
		}
		for _, ch := range mapping[digits[i]-'0'] {
			path = append(path, byte(ch))
			dfs(i+1, path)
			//恢复现场
			path = path[:len(path)-1]
		}
	}
	if len(digits) == 0 {
		return
	}
	dfs(0, []byte{})
	return
}

// 组合: 从1-n中选取k个数
func combine(n int, k int) (ans [][]int) {
	//i代表当前从哪开始选择数字
	var dfs func(start int, path []int)
	dfs = func(start int, path []int) {
		if len(path) == k {
			ans = append(ans, append([]int{}, path...))
			return
		}
		//从start开始向下选择
		for i := start; i <= n; i++ {
			path = append(path, i)
			dfs(i+1, path)
			path = path[:len(path)-1]
		}
	}
	dfs(1, []int{})
	return
}

// 子集问题
func subsets(nums []int) (ans [][]int) {
	var dfs func(start int, path []int)
	dfs = func(start int, path []int) {
		//采用枚举选择的数字时 每一步都是答案
		//而采用选或者不选模式时 一直到最后才是答案 因为每一步都可以不选
		ans = append(ans, append([]int{}, path...))
		if start == len(nums) {
			return
		}
		//选或者不选
		//dfs(start+1, path)
		//path = append(path, nums[start])
		//dfs(start+1, path)
		//path = path[:len(path)-1]
		//枚举选择的数字
		for i := start; i < len(nums); i++ {
			path = append(path, nums[i])
			dfs(i+1, path)
			path = path[:len(path)-1]
		}
	}
	dfs(0, []int{})
	return
}

func isPalindrome(str string) bool {
	for i := 0; i < len(str)/2; i++ {
		if str[i] != str[len(str)-i-1] {
			return false
		}
	}
	return true
}

// 分割回文串
// 每一个位置都有切割或者不切割两种选择
func partition2(s string) (ans [][]string) {
	var dfs func(pre, i int, path []string)
	dfs = func(pre, i int, path []string) {
		if i == len(s) {
			ans = append(ans, append([]string{}, path...))
			return
		}
		//如果是最后一个位置 必须切割
		if i < len(s)-1 {
			dfs(pre, i+1, path)
		}
		//切割 先判断能否构成回文串
		if isPalindrome(s[pre : i+1]) {
			path = append(path, s[pre:i+1])
			dfs(i+1, i+1, path)
			path = path[:len(path)-1]
		}
	}
	dfs(0, 0, []string{})
	return
}

// 枚举每一个切割的位置
func partition3(s string) (ans [][]string) {
	var dfs func(start int, path []string)
	dfs = func(start int, path []string) {
		if start == len(s) {
			ans = append(ans, append([]string{}, path...))
			return
		}
		//枚举每一个切割的位置
		for i := start; i < len(s); i++ {
			t := s[start : i+1]
			if isPalindrome(t) {
				path = append(path, t)
				dfs(i+1, path)
				path = path[:len(path)-1]
			}
		}
	}
	dfs(0, []string{})
	return
}
