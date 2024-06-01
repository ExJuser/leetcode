package main

import (
	"math"
	"math/bits"
	"slices"
	"strings"
)

func pathSum1(root *TreeNode, targetSum int) (ans [][]int) {
	var dfs func(node *TreeNode, sum int, path []int)
	dfs = func(node *TreeNode, sum int, path []int) {
		if node == nil {
			return
		}
		path = append(path, node.Val)
		sum += node.Val
		if node.Left == nil && node.Right == nil {
			if sum == targetSum {
				ans = append(ans, append([]int{}, path...))
			}
		}
		dfs(node.Left, sum, path)
		dfs(node.Right, sum, path)
	}
	dfs(root, 0, []int{})
	return
}

func hasPathSum1(root *TreeNode, targetSum int) bool {
	var dfs func(node *TreeNode, sum int) bool
	dfs = func(node *TreeNode, sum int) bool {
		if node == nil {
			return false
		}
		sum += node.Val
		if node.Left == nil && node.Right == nil && sum == targetSum {
			return true
		}
		if dfs(node.Left, sum) || dfs(node.Right, sum) {
			return true
		}
		return false
	}
	return dfs(root, 0)
}
func kConcatenationMaxSum1(arr []int, k int) int {
	var maxSubArray func(nums []int) int
	maxSubArray = func(nums []int) int {
		ans := nums[0]
		for i := 1; i < len(nums); i++ {
			nums[i] = max(nums[i], (nums[i-1]+nums[i])%(1e9+7))
			ans = max(ans, nums[i])
		}
		return max(ans, 0)
	}
	newArr := make([]int, 0, 3*len(arr))
	for ; k > 0; k-- {
		newArr = append(newArr, arr...)
	}
	return maxSubArray(newArr)
}
func removeElement1(nums []int, val int) int {
	n := len(nums)
	for i := 0; i < n; i++ {
		for i < n && nums[i] == val {
			nums[i], nums[n-1] = nums[n-1], nums[i]
			n--
		}
	}
	return n
}
func removeDuplicates1(nums []int) int {
	index, times := 1, 1
	for i := 1; i < len(nums); i++ {
		if nums[i] == nums[index-1] {
			times++
			if times <= 2 {
				nums[index] = nums[i]
				index++
			}
		} else {
			times = 1
			nums[index] = nums[i]
			index++
		}
	}
	return index
}
func plusOne(digits []int) []int {
	carry := 0
	for i := len(digits) - 1; i >= 0; i-- {
		sum := digits[i] + carry + 1
		carry = sum / 10
		digits[i] = sum % 10
	}
	if carry != 0 {
		digits = append([]int{1}, digits...)
	}
	return digits
}
func hammingWeight(n int) int {
	return bits.OnesCount32(uint32(n))
}
func reverseBits(num uint32) uint32 {
	return bits.Reverse32(num)
}
func asteroidCollision(asteroids []int) []int {
	stack := make([]int, 0, len(asteroids))
	for i := 0; i < len(asteroids); i++ {
		crash := false
		for len(stack) > 0 && stack[len(stack)-1] > 0 && asteroids[i] < 0 {
			if Abs(stack[len(stack)-1]) < Abs(asteroids[i]) {
				stack = stack[:len(stack)-1]
			} else {
				if stack[len(stack)-1]+asteroids[i] == 0 {
					stack = stack[:len(stack)-1]
				}
				crash = true
				break
			}
		}
		if !crash {
			stack = append(stack, asteroids[i])
		}
	}
	return stack
}
func reverseVowels(s string) string {
	bytes := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		if strings.Contains("aoieu", strings.ToLower(s[i:i+1])) {
			bytes = append(bytes, s[i])
		}
	}
	slices.Reverse(bytes)
	newS := []byte(s)
	index := 0
	for i := 0; i < len(newS); i++ {
		if strings.Contains("aoieu", strings.ToLower(string(newS[i]))) {
			newS[i] = bytes[index]
			index++
		}
	}
	return string(newS)
}
func distributeCandies(n int, limit int) int {
	ans := 0
	for i := 0; i <= limit; i++ {
		for j := 0; j <= limit; j++ {
			k := n - i - j
			if k >= 0 && k <= limit {
				ans++
			}
		}
	}
	return ans
}

// 二叉树最大宽度
func widthOfBinaryTree(root *TreeNode) (ans int) {
	type Pair struct {
		index int
		node  *TreeNode
	}
	queue := make([]Pair, 0)
	queue = append(queue, Pair{index: 1, node: root})
	for len(queue) > 0 {
		size := len(queue)
		ans = max(ans, queue[size-1].index-queue[0].index+1)
		for i := 0; i < size; i++ {
			temp := queue[0]
			queue = queue[1:]
			if temp.node.Left != nil {
				queue = append(queue, Pair{index: temp.index * 2, node: temp.node.Left})
			}
			if temp.node.Right != nil {
				queue = append(queue, Pair{index: temp.index*2 + 1, node: temp.node.Right})
			}
		}
	}
	return
}
func maxPathSum1(root *TreeNode) int {
	//每一个节点的最大路径和为左子树+右子树
	//但是返回给上层节点的是两者中的最大值
	ans := math.MinInt
	var dfs func(node *TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		left := dfs(node.Left)
		right := dfs(node.Right)
		ans = max(ans, node.Val, node.Val+left, node.Val+right, node.Val+left+right)
		return max(left, right, 0) + node.Val
	}
	dfs(root)
	return ans
}

func invertTree1(root *TreeNode) *TreeNode {
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		node.Left, node.Right = node.Right, node.Left
		dfs(node.Left)
		dfs(node.Right)
	}
	dfs(root)
	return root
}

func isCompleteTree(root *TreeNode) bool {
	sequence := make([]int, 0, 1000)
	var dfs func(node *TreeNode, code int)
	dfs = func(node *TreeNode, code int) {
		if node == nil {
			return
		}
		sequence = append(sequence, code)
		dfs(node.Left, code*2)
		dfs(node.Right, code*2+1)
	}
	dfs(root, 1)
	slices.Sort(sequence)
	for i := 0; i < len(sequence); i++ {
		if sequence[i] != i+1 {
			return false
		}
	}
	return true
}

// 寻找第k大 小根堆
//
//	func findTargetNode(root *TreeNode, cnt int) int {
//		hp := &IntHeap{}
//		var dfs func(node *TreeNode)
//		dfs = func(node *TreeNode) {
//			if node == nil {
//				return
//			}
//			heap.Push(hp, node.Val)
//			if hp.Len() > cnt {
//				heap.Pop(hp)
//			}
//			dfs(node.Left)
//			dfs(node.Right)
//		}
//		dfs(root)
//		return heap.Pop(hp).(int)
//	}
func findTargetNode(root *TreeNode, cnt int) int {
	//在当前子树中寻找第cnt大的数字
	//如果右子树的节点数量>cnt-1:递归右子树寻找第cnt大的数字
	//如果右子树的节点数量=cnt-1:返回当前节点
	//如果右子树的节点数量<cnt-1:递归左子树寻找第cnt-right-1大的数字
	//数量、答案
	var dfsNodeCount func(node *TreeNode) int
	var dfsFindTarget func(node *TreeNode, cnt int) int
	dfsNodeCount = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		return dfsNodeCount(node.Left) + dfsNodeCount(node.Right) + 1
	}
	dfsFindTarget = func(node *TreeNode, cnt int) int {
		if node == nil {
			return -1
		}
		rightNodeCount := dfsNodeCount(node.Right)
		ans := 0
		if rightNodeCount > cnt-1 {
			ans = dfsFindTarget(node.Right, cnt)
		} else if rightNodeCount == cnt-1 {
			ans = node.Val
		} else {
			ans = dfsFindTarget(node.Left, cnt-rightNodeCount-1)
		}
		return ans
	}
	return dfsFindTarget(root, cnt)
}
