package main

import (
	"math"
	"sort"
	"strings"
)

// 139. 单词拆分
func wordBreak(s string, wordDict []string) bool {
	dp := make([]bool, len(s))
	for _, word := range wordDict {
		if strings.HasPrefix(s, word) {
			dp[len(word)-1] = true
		}
	}
	for i := 0; i < len(dp); i++ {
		if dp[i] {
			for j := 0; j < len(wordDict); j++ {
				if strings.HasPrefix(s[i+1:], wordDict[j]) {
					dp[i+len(wordDict[j])] = true
				}
			}
		}
	}
	return dp[len(s)-1]
}

// 300. 最长递增子序列 非优化版本
//
//	func lengthOfLIS(nums []int) int {
//		dp := make([]int, len(nums))
//		for i := 0; i < len(dp); i++ {
//			dp[i] = 1
//		}
//		var ans = 1
//		for i := 1; i < len(dp); i++ {
//			for j := 0; j < i; j++ {
//				if nums[i] > nums[j] {
//					dp[i] = max(dp[i], dp[j]+1)
//					ans = max(ans, dp[i])
//				}
//			}
//		}
//		return ans
//	}
//
// 300. 最长递增子序列 优化版本
func lengthOfLIS(nums []int) int {
	sequence := make([]int, 0, len(nums))
	for _, num := range nums {
		//找到插入位置
		index := sort.SearchInts(sequence, num)
		if index == len(sequence) {
			sequence = append(sequence, num)
		} else {
			sequence[index] = num
		}
	}
	return len(sequence)
}

// 152. 乘积最大子数组
func maxProduct(nums []int) int {
	dp := make([][2]float64, len(nums))
	//最小
	dp[0][0] = float64(nums[0])
	//最大
	dp[0][1] = float64(nums[0])
	ans := float64(nums[0])
	for i := 1; i < len(nums); i++ {
		dp[i][0] = min(dp[i-1][0]*float64(nums[i]), dp[i-1][1]*float64(nums[i]), float64(nums[i]))
		dp[i][1] = max(dp[i-1][0]*float64(nums[i]), dp[i-1][1]*float64(nums[i]), float64(nums[i]))
		ans = max(ans, dp[i][1])
	}
	return int(ans)
}

// 416. 分割等和子集
func canPartition(nums []int) bool {
	var sum int
	for _, num := range nums {
		sum += num
	}
	if sum%2 != 0 {
		return false
	}
	target := sum / 2
	//dp[j] 容量为j的背包 最大价值
	dp := make([]int, target+1)
	for i := 0; i < len(nums); i++ {
		for j := target; j >= 0; j-- {
			if j >= nums[i] {
				dp[j] = max(dp[j], dp[j-nums[i]]+nums[i])
			}
		}
	}
	return dp[target] == target
}

// 994. 腐烂的橘子
func orangesRotting(grid [][]int) int {
	direction := [][]int{
		{0, 1}, {0, -1}, {1, 0}, {-1, 0},
	}
	//统计新鲜的橘子的个数 腐烂的橘子加入队列
	queue := make([][2]int, 0)
	var freshCount int
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == 1 {
				freshCount++
			} else if grid[i][j] == 2 {
				queue = append(queue, [2]int{i, j})
			}
		}
	}
	var ans int
	if freshCount == 0 {
		return ans
	}
	for len(queue) > 0 {
		ans++
		size := len(queue)
		for i := 0; i < size; i++ {
			rot := queue[0]
			queue = queue[1:]
			rotX, rotY := rot[0], rot[1]
			for _, d := range direction {
				x, y := rotX+d[0], rotY+d[1]
				if x >= 0 && x < len(grid) && y >= 0 && y < len(grid[0]) && grid[x][y] == 1 {
					grid[x][y] = 2
					freshCount--
					if freshCount == 0 {
						return ans
					}
					queue = append(queue, [2]int{x, y})
				}
			}
		}
	}
	return -1
}

// 207. 课程表 拓扑排序
func canFinish(numCourses int, prerequisites [][]int) bool {
	graph := make([][]int, numCourses)
	inDegrees := make(map[int]int)
	for _, pre := range prerequisites {
		preCourse, course := pre[1], pre[0]
		graph[preCourse] = append(graph[preCourse], course)
		inDegrees[course]++
	}
	queue := make([]int, 0, numCourses)
	for i := 0; i < numCourses; i++ {
		if inDegrees[i] == 0 {
			queue = append(queue, i)
		}
	}
	for len(queue) > 0 {
		x := queue[0]
		queue = queue[1:]
		for _, e := range graph[x] {
			inDegrees[e]--
			if inDegrees[e] == 0 {
				delete(inDegrees, e)
				queue = append(queue, e)
			}
		}
	}
	return len(inDegrees) == 0
}

// 32. 最长有效括号 先用栈模拟一遍得到非法位置置为1 然后找到最长的0序列
func longestValidParentheses(s string) int {
	stack := make([]int, 0, len(s))
	for i, ch := range s {
		if ch == '(' { //左括号直接入栈
			stack = append(stack, i)
		} else {
			//右括号
			if len(stack) != 0 && s[stack[len(stack)-1]] == '(' {
				stack = stack[:len(stack)-1]
			} else {
				stack = append(stack, i)
			}
		}
	}
	sequence := make([]int, len(s))
	for i := 0; i < len(stack); i++ {
		sequence[stack[i]] = 1
	}
	//找出最长的0序列
	var length int
	var ans int
	for i := 0; i < len(sequence); i++ {
		if sequence[i] == 0 {
			length++
			ans = max(ans, length)
		} else {
			length = 0
		}
	}
	return ans
}

// 905. 按奇偶排序数组
func sortArrayByParity(nums []int) []int {
	var index int
	for i := 0; i < len(nums); i++ {
		if nums[i]%2 == 0 {
			nums[index], nums[i] = nums[i], nums[index]
			index++
		}
	}
	return nums
}

// 162. 寻找峰值
func findPeakElement(nums []int) int {
	if len(nums) == 1 {
		return 0
	}
	//左右边界
	if nums[0] > nums[1] {
		return 0
	} else if nums[len(nums)-1] > nums[len(nums)-2] {
		return len(nums) - 1
	}

	left, right := 0, len(nums)-1
	for left <= right {
		mid := (right-left)/2 + left
		//大于左而且大于右
		if (mid-1 < 0 || nums[mid] > nums[mid-1]) && (mid+1 >= len(nums) || nums[mid] > nums[mid+1]) {
			return mid
		}
		//大于左
		if mid-1 < 0 || nums[mid] > nums[mid-1] {
			left = mid + 1
		} else { //直接else即可 因为只要两个边界不符合 中间一定存在
			right = mid - 1
		}
	}
	return -1
}

// 915. 分割数组
func partitionDisjoint(nums []int) int {
	rightMin := make([]int, len(nums))
	rightMin[len(nums)-1] = math.MaxInt
	for i := len(nums) - 2; i >= 0; i-- {
		rightMin[i] = min(rightMin[i+1], nums[i+1])
	}
	var curMax int
	for i := 0; i < len(nums); i++ {
		curMax = max(curMax, nums[i])
		if curMax <= rightMin[i] {
			return i + 1
		}
	}
	return -1
}

func numIslands(grid [][]byte) int {
	directions := [][]int{
		{0, 1}, {0, -1}, {-1, 0}, {1, 0},
	}
	var dfs func(i, j int)
	dfs = func(i, j int) {
		if i < 0 || i >= len(grid) || j < 0 || j >= len(grid[i]) || grid[i][j] != '1' {
			return
		}
		grid[i][j] = '0'
		for _, d := range directions {
			ii, jj := i+d[0], j+d[1]
			dfs(ii, jj)
		}
	}
	var ans int
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == '1' {
				ans++
				dfs(i, j)
			}
		}
	}
	return ans
}

func maxAreaOfIsland(grid [][]int) int {
	directions := [][]int{
		{0, 1}, {0, -1}, {-1, 0}, {1, 0},
	}
	var dfs func(i, j int) int
	dfs = func(i, j int) int {
		if i < 0 || i >= len(grid) || j < 0 || j >= len(grid[i]) || grid[i][j] != 1 {
			return 0
		}
		grid[i][j] = 0
		var count = 1
		for _, d := range directions {
			ii, jj := i+d[0], j+d[1]
			count += dfs(ii, jj)
		}
		return count
	}
	var ans int
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == 1 {
				ans = max(ans, dfs(i, j))
			}
		}
	}
	return ans
}

//func sortArray(nums []int) []int {
//	var helper func(left, right int)
//	helper = func(left, right int) {
//		if left >= right {
//			return
//		}
//		i, j := left, right
//		pivot := nums[rand.IntN(right-left+1)+left]
//		for i <= j {
//			for nums[i] < pivot {
//				i++
//			}
//			for nums[j] > pivot {
//				j--
//			}
//			if i <= j {
//				nums[i], nums[j] = nums[j], nums[i]
//				i++
//				j--
//			}
//		}
//		helper(left, j)
//		helper(i, right)
//	}
//	helper(0, len(nums)-1)
//	return nums
//}

// 迭代实现
//func levelOrder(root *TreeNode) (ans [][]int) {
//	if root == nil {
//		return
//	}
//	queue := make([]*TreeNode, 0)
//	queue = append(queue, root)
//	for len(queue) > 0 {
//		size := len(queue)
//		level := make([]int, 0, size)
//		for i := 0; i < size; i++ {
//			x := queue[0]
//			queue = queue[1:]
//			level = append(level, x.Val)
//			if x.Left != nil {
//				queue = append(queue, x.Left)
//			}
//			if x.Right != nil {
//				queue = append(queue, x.Right)
//			}
//		}
//		ans = append(ans, level)
//	}
//	return
//}

// 递归实现
func levelOrder(root *TreeNode) (ans [][]int) {
	var dfs func(node *TreeNode, depth int)
	dfs = func(node *TreeNode, depth int) {
		if node == nil {
			return
		}
		if depth == len(ans) {
			ans = append(ans, []int{node.Val})
		} else {
			ans[depth] = append(ans[depth], node.Val)
		}
		dfs(node.Left, depth+1)
		dfs(node.Right, depth+1)
	}
	dfs(root, 0)
	return
}

// 515. 在每个树行中找最大值
func largestValues(root *TreeNode) (ans []int) {
	var dfs func(node *TreeNode, depth int)
	dfs = func(node *TreeNode, depth int) {
		if node == nil {
			return
		}
		if depth == len(ans) {
			ans = append(ans, node.Val)
		} else {
			ans[depth] = max(ans[depth], node.Val)
		}
		dfs(node.Left, depth+1)
		dfs(node.Right, depth+1)
	}
	dfs(root, 0)
	return
}

// 344. 反转字符串
func reverseString(s []byte) {
	for i := 0; i < len(s)/2; i++ {
		s[i], s[len(s)-i-1] = s[len(s)-i-1], s[i]
	}
}

// 82. 删除排序链表中的重复元素 II 只留下不重复的数字
//func deleteDuplicates(head *ListNode) *ListNode {
//	dummy := &ListNode{Next: head}
//
//	cur := dummy
//	for cur.Next != nil && cur.Next.Next != nil {
//		val := cur.Next.Val
//		if cur.Next.Next.Val == val {
//			for cur.Next != nil && cur.Next.Val == val {
//				cur.Next = cur.Next.Next
//			}
//		} else {
//			cur = cur.Next
//		}
//	}
//
//	return dummy.Next
//}

// 82. 删除排序链表中的重复元素 II 递归更简单一点
func deleteDuplicates2(head *ListNode) *ListNode {
	var dfs func(node *ListNode) *ListNode
	dfs = func(node *ListNode) *ListNode {
		if node == nil || node.Next == nil {
			return node
		}
		if node.Val == node.Next.Val {
			val := node.Val
			for node.Next != nil && node.Next.Val == val {
				node = node.Next
			}
			return dfs(node.Next)
		} else {
			node.Next = dfs(node.Next)
			return node
		}
	}
	return dfs(head)
}

// 83. 删除排序链表中的重复元素 也可以用递归解决
//func deleteDuplicates(head *ListNode) *ListNode {
//	var dfs func(node *ListNode) *ListNode
//	dfs = func(node *ListNode) *ListNode {
//		if node == nil || node.Next == nil {
//			return node
//		}
//		cur := node
//		if node.Val == node.Next.Val {
//			//向后遍历到第一个不相等的
//			val := node.Val
//			for cur.Next != nil && cur.Next.Val == val {
//				cur = cur.Next
//			}
//		}
//		node.Next = dfs(cur.Next)
//		return node
//	}
//	return dfs(head)
//}

// 保留一个 画图！
//
//	func deleteDuplicates(head *ListNode) *ListNode {
//		dummy := &ListNode{Next: head}
//		for cur := head; cur != nil; cur = cur.Next {
//			p := cur
//			for p != nil && p.Val == cur.Val {
//				p = p.Next
//			}
//			cur.Next = p
//		}
//		return dummy.Next
//	}
//
// 82. 删除排序链表中的重复元素 II
func deleteDuplicates(head *ListNode) *ListNode {
	dummy := &ListNode{Next: head}
	cur := dummy
	for cur.Next != nil && cur.Next.Next != nil {
		if cur.Next.Val == cur.Next.Next.Val {
			p := cur
			for p.Next != nil && p.Next.Val == cur.Next.Val {
				p = p.Next
			}
			cur.Next = p.Next
		} else {
			cur = cur.Next
		}
	}
	return dummy.Next
}
