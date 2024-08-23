package main

import (
	"container/heap"
	"math"
	"math/rand/v2"
	"slices"
	"sort"
	"strings"
	"unicode"
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

// 25. K 个一组翻转链表
//
//	func reverseKGroup(head *ListNode, k int) *ListNode {
//		dummy := &ListNode{Next: head}
//		//首先需要获取链表的长度
//		var length int
//		for cur := head; cur != nil; cur = cur.Next {
//			length++
//		}
//		var pre *ListNode
//		cur := head
//		temp := dummy
//		for i := 0; i < length/k; i++ {
//			for j := 0; j < k; j++ {
//				nxt := cur.Next
//				cur.Next = pre
//				cur, pre = nxt, cur
//			}
//			newTemp := temp.Next
//			temp.Next.Next = cur
//			temp.Next = pre
//			temp = newTemp
//		}
//		return dummy.Next
//	}
//
// 25. K 个一组翻转链表 递归
func reverseKGroup(head *ListNode, k int) *ListNode {
	var dfs func(node *ListNode) *ListNode
	var length func(node *ListNode) int
	dfs = func(node *ListNode) *ListNode {
		//如果剩余的节点数量不足k个 直接返回
		if length(node) < k {
			return node
		}

		cur := node
		temp := node
		var pre *ListNode
		for i := 0; i < k; i++ {
			nxt := cur.Next
			cur.Next = pre
			pre, cur = cur, nxt
		}
		temp.Next = dfs(cur)
		return pre
	}
	length = func(node *ListNode) int {
		var ans int
		for cur := node; cur != nil; cur = cur.Next {
			ans++
		}
		return ans
	}
	return dfs(head)
}

// 572. 另一棵树的子树 写的很烂
//
//	func isSubtree(root *TreeNode, subRoot *TreeNode) bool {
//		var isSameTree func(treeA, treeB *TreeNode) bool
//		isSameTree = func(treeA, treeB *TreeNode) bool {
//			if treeA == nil || treeB == nil {
//				return treeA == treeB
//			}
//			return treeA.Val == treeB.Val && isSameTree(treeA.Left, treeB.Left) && isSameTree(treeA.Right, treeB.Right)
//		}
//		var dfs func(root *TreeNode, subRoot *TreeNode) bool
//		dfs = func(root *TreeNode, subRoot *TreeNode) bool {
//			if root == nil {
//				return false
//			}
//			if isSameTree(root, subRoot) {
//				return true
//			}
//			if dfs(root.Left, subRoot) || dfs(root.Right, subRoot) {
//				return true
//			}
//			return false
//		}
//		return dfs(root, subRoot)
//	}
//
// 572. 另一棵树的子树
func isSubtree(root *TreeNode, subRoot *TreeNode) bool {
	var isSameTree func(treeA, treeB *TreeNode) bool
	isSameTree = func(treeA, treeB *TreeNode) bool {
		if treeA == nil || treeB == nil {
			return treeA == treeB
		}
		return treeA.Val == treeB.Val && isSameTree(treeA.Left, treeB.Left) && isSameTree(treeA.Right, treeB.Right)
	}
	var dfs func(node *TreeNode) bool
	dfs = func(node *TreeNode) bool {
		if node == nil {
			return false
		}
		if node.Val == subRoot.Val && isSameTree(node, subRoot) {
			return true
		}
		return dfs(node.Left) || dfs(node.Right)
	}
	return dfs(root)
}

// 第一种：快速选择算法
func findKthLargest(nums []int, k int) int {
	var helper func(left, right, k int) int
	helper = func(left, right, k int) int {
		if left >= right {
			return nums[k]
		}
		pivot := nums[rand.IntN(right-left+1)+left]
		i, j := left, right
		for i <= j {
			for nums[i] < pivot {
				i++
			}
			for nums[j] > pivot {
				j--
			}
			if i <= j {
				nums[i], nums[j] = nums[j], nums[i]
				i++
				j--
			}
		}
		if k <= j {
			return helper(left, j, k)
		} else {
			return helper(i, right, k)
		}
	}
	return helper(0, len(nums)-1, len(nums)-k)
}

// 堆排序 最小堆
func findKthLargest2(nums []int, k int) int {
	hp := &minHeap{}
	for _, num := range nums {
		heap.Push(hp, num)
		if hp.Len() > k {
			heap.Pop(hp)
		}
	}
	return (*hp)[0]
}

// 209. 长度最小的子数组
func minSubArrayLen(target int, nums []int) int {
	var ans = len(nums) + 1
	var left, sum int
	for right := 0; right < len(nums); right++ {
		sum += nums[right]
		for ; sum >= target; left++ {
			ans = min(ans, right-left+1)
			sum -= nums[left]
		}
	}
	if ans == len(nums)+1 {
		return 0
	}
	return ans
}

// 335. 路径交叉 暴力超时
func isSelfCrossing(distance []int) bool {
	directions := [][]int{
		{0, 1}, {-1, 0}, {0, -1}, {1, 0},
	}
	oriX, oriY := 0, 0
	set := map[[2]int]struct{}{[2]int{0, 0}: {}}
	for i := 0; i < len(distance); i++ {
		d := directions[i%4]
		for j := 0; j < distance[i]; j++ {
			x, y := oriX+d[0], oriY+d[1]
			if _, ok := set[[2]int{x, y}]; ok { //交叉
				return true
			}
			set[[2]int{x, y}] = struct{}{}
			oriX, oriY = x, y
		}
	}
	return false
}

// 435. 无重叠区间
func eraseOverlapIntervals(intervals [][]int) int {
	slices.SortFunc(intervals, func(a, b []int) int {
		return a[0] - b[0]
	})
	var ans int
	pre := intervals[0]
	for i := 1; i < len(intervals); i++ {
		cur := intervals[i]
		if cur[0] >= pre[1] { //没有重叠
			pre = cur
		} else { //有重叠部分 这两个必须移除一个 那么移除哪一个？
			//移除右边更长的那一个
			ans++
			if cur[1] < pre[1] {
				pre = cur
			}
		}
	}
	return ans
}

// 先统计一遍0,1,2分别有多少个
//
//	func sortColors(nums []int) {
//		var zero, one int
//		for _, num := range nums {
//			if num == 0 {
//				zero++
//			} else if num == 1 {
//				one++
//			}
//		}
//		var zeroIndex, oneIndex, twoIndex = 0, zero, zero + one
//		for i := 0; i < len(nums); {
//			if nums[i] == 0 && zeroIndex < zero {
//				nums[zeroIndex], nums[i] = nums[i], nums[zeroIndex]
//				zeroIndex++
//			} else if nums[i] == 1 && oneIndex < zero+one {
//				nums[oneIndex], nums[i] = nums[i], nums[oneIndex]
//				oneIndex++
//			} else if nums[i] == 2 && twoIndex < len(nums) {
//				nums[twoIndex], nums[i] = nums[i], nums[twoIndex]
//				twoIndex++
//			} else {
//				i++
//			}
//		}
//	}
//
// 75. 颜色分类
func sortColors(nums []int) {
	zeroIndex, twoIndex := 0, len(nums)-1
	for i := 0; i <= twoIndex; {
		if nums[i] == 0 {
			nums[zeroIndex], nums[i] = nums[i], nums[zeroIndex]
			zeroIndex++
			i++
		} else if nums[i] == 2 {
			nums[twoIndex], nums[i] = nums[i], nums[twoIndex]
			twoIndex--
		} else {
			i++
		}
	}
}

//	func lengthOfLIS(nums []int) int {
//		sequence := make([]int, 0, len(nums))
//		for _, num := range nums {
//			//找到插入位置
//			index := sort.SearchInts(sequence, num)
//			if index == len(sequence) {
//				sequence = append(sequence, num)
//			} else {
//				sequence[index] = num
//			}
//		}
//		return len(sequence)
//	}
//

func merge(nums1 []int, m int, nums2 []int, n int) {
	index := m + n - 1
	i, j := m-1, n-1
	for i >= 0 && j >= 0 {
		if nums1[i] >= nums2[j] {
			nums1[index] = nums1[i]
			i--
		} else {
			nums1[index] = nums2[j]
			j--
		}
		index--
	}
	for ; i >= 0; i-- {
		nums1[index] = nums1[i]
		index--
	}

	for ; j >= 0; j-- {
		nums1[index] = nums2[j]
		index--
	}
}

// 59. 螺旋矩阵 II 注意++ -- 和 i 的位置
func generateMatrix(n int) [][]int {
	matrix := make([][]int, n)
	for i := 0; i < n; i++ {
		matrix[i] = make([]int, n)
	}
	top, bottom := 0, n-1
	left, right := 0, n-1
	var num = 1
	for left <= right && top <= bottom {
		for i := left; i <= right; i++ {
			matrix[top][i] = num
			num++
		}
		top++
		for i := top; i <= bottom; i++ {
			matrix[i][right] = num
			num++
		}
		right--
		for i := right; i >= left; i-- {
			matrix[bottom][i] = num
			num++
		}
		bottom--
		for i := bottom; i >= top; i-- {
			matrix[i][left] = num
			num++
		}
		left++
	}
	return matrix
}

// 560. 和为 K 的子数组
func subarraySum(nums []int, k int) int {
	prefix := map[int]int{0: 1}
	var sum, ans int
	for i := 0; i < len(nums); i++ {
		sum += nums[i]
		ans += prefix[sum-k]
		prefix[sum]++
	}
	return ans
}

func minimumTotal(triangle [][]int) int {
	//可以从相同下标或者下标减一的位置过来
	n := len(triangle)
	if n == 1 {
		return triangle[0][0]
	}
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, len(triangle[i]))
	}
	dp[0][0] = triangle[0][0]
	var ans = math.MaxInt
	for i := 1; i < n; i++ {
		for j := 0; j < len(triangle[i]); j++ {
			if j >= len(dp[i-1]) {
				dp[i][j] = dp[i-1][j-1] + triangle[i][j]
			} else if j-1 < 0 {
				dp[i][j] = dp[i-1][j] + triangle[i][j]
			} else {
				dp[i][j] = min(dp[i-1][j-1]+triangle[i][j], dp[i-1][j]+triangle[i][j])
			}
			if i == n-1 {
				ans = min(ans, dp[i][j])
			}
		}
	}
	return ans
}

// 125. 验证回文串 将所有大写字符转换为小写字符、并移除所有非字母数字字符
func isPalindrome(s string) bool {
	bytes := make([]byte, 0, len(s))
	for _, ch := range s {
		if unicode.IsLetter(ch) { //是字母
			bytes = append(bytes, byte(unicode.ToLower(ch)))
		} else if unicode.IsDigit(ch) { //是数字
			bytes = append(bytes, byte(ch))
		}
	}
	for i := 0; i < len(bytes); i++ {
		if bytes[i] != bytes[len(bytes)-i-1] {
			return false
		}
	}
	return true
}

// 面试题 17.15. 最长单词
func longestWord(words []string) string {
	//可以使用多次
	var combinedByOtherWords func(word string) bool
	combinedByOtherWords = func(word string) bool {
		if len(word) == 0 {
			return false
		}
		dp := make([]bool, len(word))
		for i := 0; i < len(words); i++ {
			if words[i] != word && strings.HasPrefix(word, words[i]) {
				dp[len(words[i])-1] = true
			}
		}
		for i := 0; i < len(dp); i++ {
			if dp[i] {
				for j := 0; j < len(words); j++ {
					if words[j] != word && strings.HasPrefix(word[i+1:], words[j]) {
						dp[i+len(words[j])] = true
					}
				}
			}
		}
		return dp[len(word)-1]
	}
	slices.SortFunc(words, func(a, b string) int {
		if len(a) == len(b) {
			return strings.Compare(a, b)
		}
		return len(b) - len(a)
	})
	for i := 0; i < len(words); i++ {
		if combinedByOtherWords(words[i]) {
			return words[i]
		}
	}
	return ""
}
