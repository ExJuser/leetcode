package common

import (
	"container/heap"
	"fmt"
	"math/rand/v2"
	"slices"
	"sort"
	"strconv"
)

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

// 23. 合并 K 个升序链表
func mergeKLists(lists []*ListNode) *ListNode {
	hp := &NodeHeap{}
	for _, list := range lists {
		if list != nil {
			heap.Push(hp, list)
		}
	}
	dummy := &ListNode{}
	p := dummy
	for hp.Len() != 0 {
		node := heap.Pop(hp).(*ListNode)
		p.Next = &ListNode{Val: node.Val}
		p = p.Next
		if node.Next != nil {
			heap.Push(hp, node.Next)
		}
	}
	return dummy.Next
}

// 69. x 的平方根
func mySqrt(x int) int {
	left, right := 0, x
	ans := 0
	for left <= right {
		mid := (right-left)/2 + left
		if mid*mid <= x {
			ans = mid
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	return ans
}

// 34. 在排序数组中查找元素的第一个和最后一个位置
func searchRangeBinary(nums []int, target int) []int {
	//先找到第一个
	left, right := 0, len(nums)-1
	first := -1
	for left <= right {
		mid := (right-left)/2 + left
		if nums[mid] >= target {
			first = mid
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	if first == -1 || nums[first] != target {
		return []int{-1, -1}
	}
	return []int{first, sort.SearchInts(nums[first:], target+1) + first - 1}
}

// 141. 环形链表
func hasCycle(head *ListNode) bool {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			return true
		}
	}
	return false
}

// 142. 环形链表 II
func detectCycle(head *ListNode) *ListNode {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			for p := head; p != slow; {
				p = p.Next
				slow = slow.Next
			}
			return slow
		}
	}
	return nil
}

// 198. 打家劫舍
func rob(nums []int) int {
	if len(nums) <= 2 {
		return slices.Max(nums)
	}
	dp1, dp2 := nums[0], max(nums[0], nums[1])
	for i := 2; i < len(nums); i++ {
		dp1, dp2 = dp2, max(dp1+nums[i], dp2)
	}
	return dp2
}

func twoSum(nums []int, target int) []int {
	mp := make(map[int]int, len(nums))
	for i, num := range nums {
		if j, ok := mp[target-num]; ok {
			return []int{i, j}
		} else {
			mp[num] = i
		}
	}
	return nil
}

// 200. 岛屿数量 dfs版本
//
//	func numIslands(grid [][]byte) (ans int) {
//		var dfs func(i, j int)
//		dfs = func(i, j int) {
//			if i < 0 || i >= len(grid) || j < 0 || j >= len(grid[0]) {
//				return
//			}
//			if grid[i][j] == '1' {
//				grid[i][j] = '0'
//				dfs(i+1, j)
//				dfs(i-1, j)
//				dfs(i, j-1)
//				dfs(i, j+1)
//			}
//		}
//		for i := 0; i < len(grid); i++ {
//			for j := 0; j < len(grid[0]); j++ {
//				if grid[i][j] == '1' {
//					ans++
//					dfs(i, j)
//				}
//			}
//		}
//		return
//	}
//
// 200. 岛屿数量 bfs版本
//func numIslands(grid [][]byte) (ans int) {
//	var bfs func(i, j int)
//	bfs = func(i, j int) {
//		if i < 0 || j < 0 || i >= len(grid) || j >= len(grid[0]) {
//			return
//		}
//		if grid[i][j] == '1' {
//			grid[i][j] = '0'
//			queue := [][2]int{{i, j}}
//			for len(queue) > 0 {
//				x, y := queue[0][0], queue[0][1]
//				queue = queue[1:]
//				bfs(x-1, y)
//				bfs(x+1, y)
//				bfs(x, y-1)
//				bfs(x, y+1)
//			}
//		}
//	}
//	for i := 0; i < len(grid); i++ {
//		for j := 0; j < len(grid[0]); j++ {
//			if grid[i][j] == '1' {
//				ans++
//				bfs(i, j)
//			}
//		}
//	}
//	return ans
//}

// 80. 删除有序数组中的重复项 II
func removeDuplicates(nums []int) int {
	var count, index int
	for i := 0; i < len(nums); i++ {
		if i == 0 || nums[i] != nums[i-1] {
			count = 1
		} else {
			count++
		}
		if count <= 2 {
			nums[index] = nums[i]
			index++
		}
	}
	return index
}

// 33. 搜索旋转排序数组
func search(nums []int, target int) int {
	/**
	两段有序序列分情况讨论
	*/
	left, right := 0, len(nums)-1
	for left <= right {
		mid := (right-left)/2 + left
		if nums[mid] == target {
			return mid
		}
		if nums[mid] >= nums[left] {
			if target < nums[left] || target > nums[mid] {
				left = mid + 1
			} else {
				right = mid - 1
			}
		} else {
			if target > nums[right] || target < nums[mid] {
				right = mid - 1
			} else {
				left = mid + 1
			}
		}
	}
	return -1
}

// 39. 组合总和 可以重复选取
func combinationSum(candidates []int, target int) (ans [][]int) {
	var dfs func(index, sum int, path []int)
	dfs = func(index, sum int, path []int) {
		if sum > target {
			return
		}
		if index == len(candidates) {
			if sum == target {
				ans = append(ans, append([]int{}, path...))
			}
			return
		}
		//选或者不选
		path = append(path, candidates[index])
		dfs(index, sum+candidates[index], path)
		path = path[:len(path)-1]

		dfs(index+1, sum, path)
	}
	dfs(0, 0, []int{})
	return
}

// 40. 组合总和 II 只能选择一次
func combinationSum2(candidates []int, target int) (ans [][]int) {
	used := make([]bool, len(candidates))
	slices.Sort(candidates)
	var dfs func(index, sum int, path []int)
	dfs = func(index, sum int, path []int) {
		if sum >= target {
			if sum == target {
				ans = append(ans, append([]int{}, path...))
			}
			return
		}
		for i := index; i < len(candidates); i++ {
			if (i == index || candidates[i] != candidates[i-1]) && !used[i] {
				used[i] = true
				path = append(path, candidates[i])
				dfs(i+1, sum+candidates[i], path)
				used[i] = false
				path = path[:len(path)-1]
			}
		}
	}
	dfs(0, 0, []int{})
	return
}

// 215. 数组中的第K个最大元素 最小堆做法
// 时间复杂度为 n*log(k) 当k较小时近似线性复杂度
//	func findKthLargest(nums []int, k int) int {
//		hp := &IntMinHeap{}
//		for _, num := range nums {
//			heap.Push(hp, num)
//			if hp.Len() > k {
//				heap.Pop(hp)
//			}
//		}
//		return (*hp)[0]
//	}
//

// 215. 数组中的第K个最大元素 快速排序做法
// func findKthLargest(nums []int, k int) int {
//
// }

// 104. 二叉树的最大深度
func maxDepth(root *TreeNode) int {
	var dfs func(node *TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		return max(dfs(node.Left), dfs(node.Right)) + 1
	}
	return dfs(root)
}

// 435. 无重叠区间
func eraseOverlapIntervals(intervals [][]int) int {
	slices.SortFunc(intervals, func(a, b []int) int {
		return a[0] - b[0]
	})
	pre := intervals[0]
	var ans int
	for i := 1; i < len(intervals); i++ {
		cur := intervals[i]
		if cur[0] >= pre[1] {
			pre = cur
		} else {
			ans++
			if pre[1] > cur[1] {
				pre = cur
			}
		}
	}
	return ans
}

// 22. 括号生成 回溯 右括号不能大于左括号
func generateParenthesis(n int) (ans []string) {
	var dfs func(left, right int, path []byte)
	dfs = func(left, right int, path []byte) {
		if left < right {
			return
		}
		if left == n && right == n {
			ans = append(ans, string(path))
		}
		if left < n {
			path = append(path, '(')
			dfs(left+1, right, path)
			path = path[:len(path)-1]
		}
		if right < n {
			path = append(path, ')')
			dfs(left, right+1, path)
			path = path[:len(path)-1]
		}

	}
	dfs(0, 0, []byte{})
	return
}

// 189. 轮转数组 比较巧妙的做法
func rotate(nums []int, k int) {
	k %= len(nums)
	slices.Reverse(nums[:len(nums)-k])
	slices.Reverse(nums[len(nums)-k:])
	slices.Reverse(nums)
}

// FreqHeap 存储的值 出现频率 加入时间
type FreqHeap [][3]int

func (f *FreqHeap) Len() int {
	return len(*f)
}

func (f *FreqHeap) Less(i, j int) bool {
	if (*f)[i][1] == (*f)[j][1] {
		return (*f)[i][2] > (*f)[j][2]
	}
	return (*f)[i][1] > (*f)[j][1]
}

func (f *FreqHeap) Swap(i, j int) {
	(*f)[i], (*f)[j] = (*f)[j], (*f)[i]
}

func (f *FreqHeap) Push(x any) {
	*f = append(*f, x.([3]int))
}

func (f *FreqHeap) Pop() any {
	x := (*f)[(*f).Len()-1]
	*f = (*f)[:(*f).Len()-1]
	return x
}

// FreqStack 895. 最大频率栈
type FreqStack struct {
	hp    *FreqHeap
	mp    map[int]int
	index int
}

func Constructor() FreqStack {
	return FreqStack{
		hp:    &FreqHeap{},
		index: 0,
		mp:    make(map[int]int),
	}
}

func (this *FreqStack) Push(val int) {
	this.mp[val]++
	heap.Push(this.hp, [3]int{val, this.mp[val], this.index})
	this.index++
}

func (this *FreqStack) Pop() int {
	num := heap.Pop(this.hp).([3]int)[0]
	this.mp[num]--
	return num
}

// 124. 二叉树中的最大路径和
func maxPathSum(root *TreeNode) int {
	var dfs func(node *TreeNode) int
	ans := root.Val
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		left, right := dfs(node.Left), dfs(node.Right)
		ans = max(ans, max(0, left)+max(0, right)+node.Val)
		return max(0, left, right) + node.Val
	}
	dfs(root)
	return ans
}

func quickSort(nums []int) []int {
	var quickSortHelper func(left, right int)
	quickSortHelper = func(left, right int) {
		if left >= right {
			return
		}
		i, j := left, right
		pivot := nums[rand.IntN(right-left+1)+left]
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
		quickSortHelper(left, j)
		quickSortHelper(i, right)
	}
	quickSortHelper(0, len(nums)-1)
	return nums
}

// 选择排序 每次选择一个最小的放在最前面
func selectSort(nums []int) []int {
	for i := 0; i < len(nums); i++ {
		minIndex := i
		for j := i; j < len(nums); j++ {
			if nums[j] < nums[minIndex] {
				minIndex = j
			}
		}
		nums[i], nums[minIndex] = nums[minIndex], nums[i]
	}
	return nums
}

// 冒泡排序 每一次把最大的往上冒
func bubbleSort(nums []int) []int {
	for i := 0; i < len(nums); i++ {
		for j := 0; j < len(nums)-i-1; j++ {
			if nums[j] > nums[j+1] {
				nums[j], nums[j+1] = nums[j+1], nums[j]
			}
		}
	}
	return nums
}

// 15. 三数之和 固定一个数 相向双指针 去重
//func threeSum(nums []int) (ans [][]int) {
//	slices.Sort(nums)
//	for i := 0; i < len(nums); i++ {
//		n1 := nums[i]
//		if n1 <= 0 && (i == 0 || nums[i-1] != n1) {
//			left, right := i+1, len(nums)-1
//			for left < right {
//				n2, n3 := nums[left], nums[right]
//				if n1+n2+n3 < 0 {
//					left++
//				} else if n1+n2+n3 > 0 {
//					right--
//				} else {
//					ans = append(ans, []int{n1, n2, n3})
//					for left < right && nums[left] == n2 {
//						left++
//					}
//					for left < right && nums[right] == n3 {
//						right--
//					}
//				}
//			}
//		}
//	}
//	return
//}

func levelOrder(root *TreeNode) (ans [][]int) {
	queue := make([]*TreeNode, 0)
	if root != nil {
		queue = append(queue, root)
	}
	for len(queue) > 0 {
		size := len(queue)
		level := make([]int, 0, size)
		for i := 0; i < size; i++ {
			temp := queue[0]
			level = append(level, temp.Val)
			queue = queue[1:]
			if temp.Left != nil {
				queue = append(queue, temp.Left)
			}
			if temp.Right != nil {
				queue = append(queue, temp.Right)
			}
		}
		ans = append(ans, level)
	}
	return
}

// 88. 合并两个有序数组 倒序遍历
//func merge(nums1 []int, m int, nums2 []int, n int) {
//	i, j := m-1, n-1
//	for i >= 0 && j >= 0 {
//		if nums1[i] >= nums2[j] {
//			nums1[i+j+1] = nums1[i]
//			i--
//		} else {
//			nums1[i+j+1] = nums2[j]
//			j--
//		}
//	}
//	if j >= 0 {
//		for k := 0; k <= j; k++ {
//			nums1[k] = nums2[k]
//		}
//	}
//}

func maxProfit(prices []int) int {
	//两种状态 持有和卖出
	dp := make([][2]int, len(prices))
	dp[0][0] = -prices[0]
	for i := 1; i < len(prices); i++ {
		dp[i][0] = max(dp[i-1][0], -prices[i])
		dp[i][1] = max(dp[i-1][1], dp[i-1][0]+prices[i])
	}
	return dp[len(prices)-1][1]
}

//func zigzagLevelOrder(root *TreeNode) [][]int {
//	levels := levelOrder(root)
//	for i := 0; i < len(levels); i++ {
//		if i%2 == 1 {
//			slices.Reverse(levels[i])
//		}
//	}
//	return levels
//}

// 92. 反转链表 II
func reverseBetween(head *ListNode, left int, right int) *ListNode {
	dummy := &ListNode{Next: head}
	p := dummy
	var temp *ListNode
	for i := 0; i < left; i++ {
		if i == left-1 {
			temp = p
		}
		p = p.Next
	}
	cur := p
	var pre *ListNode
	for i := left; i <= right; i++ {
		nxt := cur.Next
		cur.Next = pre
		pre, cur = cur, nxt
	}
	temp.Next.Next = cur
	temp.Next = pre
	return dummy.Next
}

// 300. 最长递增子序列 n^2 复杂度
func lengthOfLIS(nums []int) int {
	//dpi 代表到i为止的最长递增子序列
	dp := make([]int, len(nums))
	for i := 0; i < len(dp); i++ {
		dp[i] = 1
	}
	ans := 1
	for i := 1; i < len(nums); i++ {
		for j := i - 1; j >= 0; j-- {
			if nums[i] > nums[j] {
				dp[i] = max(dp[i], dp[j]+1)
				ans = max(ans, dp[i])
			}
		}
	}
	return ans
}

// 143. 重排链表
func reorderList(head *ListNode) {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	reversed := reverseList(slow.Next)
	slow.Next = nil
	q := head
	for p := reversed; p != nil; {
		nxt := p.Next
		p.Next = q.Next
		q.Next = p
		q = p.Next
		p = nxt
	}
}

// 56. 合并区间
func merge(intervals [][]int) (ans [][]int) {
	slices.SortFunc(intervals, func(a, b []int) int {
		return a[0] - b[0]
	})
	pre := intervals[0]
	for i := 1; i < len(intervals); i++ {
		cur := intervals[i]
		if cur[0] > pre[1] {
			ans = append(ans, pre)
			pre = cur
		} else {
			pre[1] = max(pre[1], cur[1])
		}
	}
	ans = append(ans, pre)
	return
}

func rotate_(matrix [][]int) {
	for i := 0; i < len(matrix); i++ {
		slices.Reverse(matrix[i])
	}
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[i]); j++ {
			matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
		}
	}
}

func pathSum(root *TreeNode, targetSum int) (ans [][]int) {
	var dfs func(node *TreeNode, sum int, path []int)
	dfs = func(node *TreeNode, sum int, path []int) {
		if node == nil {
			return
		}
		sum += node.Val
		path = append(path, node.Val)
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

// 100. 相同的树
func isSameTree(p *TreeNode, q *TreeNode) bool {
	var dfs func(node1, node2 *TreeNode) bool
	dfs = func(node1, node2 *TreeNode) bool {
		if node1 == nil || node2 == nil {
			return node1 == node2
		}
		if node1.Val != node2.Val {
			return false
		}
		return dfs(node1.Left, node2.Left) && dfs(node1.Right, node2.Right)
	}
	return dfs(p, q)
}

// 98. 验证二叉搜索树 中序遍历序列有序
func isValidBST(root *TreeNode) bool {
	var res []int
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		res = append(res, node.Val)
		dfs(node.Right)
	}
	dfs(root)
	for i := 1; i < len(res); i++ {
		if res[i] <= res[i-1] {
			return false
		}
	}
	return true
}

// 107. 二叉树的层序遍历 II
func levelOrderBottom(root *TreeNode) [][]int {
	levels := levelOrder(root)
	slices.Reverse(levels)
	return levels
}

// 1022. 从根到叶的二进制数之和
func sumRootToLeaf(root *TreeNode) int {
	var sum int
	var dfs func(node *TreeNode, path []byte)
	var pathToint func(path []byte) int
	dfs = func(node *TreeNode, path []byte) {
		if node == nil {
			return
		}
		path = append(path, byte(node.Val+'0'))
		if node.Left == nil && node.Right == nil {
			sum += pathToint(path)
			return
		}
		dfs(node.Left, path)
		dfs(node.Right, path)
	}
	pathToint = func(path []byte) int {
		i, _ := strconv.ParseInt(string(path), 2, 64)
		fmt.Println(string(path))
		return int(i)
	}
	dfs(root, []byte{})
	return sum
}

// 494. 目标和 回溯做法非常慢
func findTargetSumWays(nums []int, target int) int {
	var dfs func(index, sum int)
	var ans int
	dfs = func(index, sum int) {
		if index == len(nums) {
			if sum == target {
				ans++
			}
			return
		}
		//加法
		dfs(index+1, sum+nums[index])
		//减法
		dfs(index+1, sum-nums[index])
	}
	dfs(0, 0)
	return ans
}

func zigzagLevelOrder(root *TreeNode) (ans [][]int) {
	if root == nil {
		return
	}
	queue := make([]*TreeNode, 0)
	queue = append(queue, root)
	depth := 0
	for len(queue) > 0 {
		size := len(queue)
		level := make([]int, size)
		for i := 0; i < size; i++ {
			temp := queue[0]
			queue = queue[1:]
			if depth%2 > 0 {
				level[size-i-1] = temp.Val
			} else {
				level[i] = temp.Val
			}
			if temp.Left != nil {
				queue = append(queue, temp.Left)
			}
			if temp.Right != nil {
				queue = append(queue, temp.Right)
			}
		}
		depth++
		ans = append(ans, level)
	}
	return
}

// 918. 环形子数组的最大和
func maxSubarraySumCircular(nums []int) int {
	//第一种情况 最大和在数组中间，即普通的子数组最大和问题
	if len(nums) == 1 {
		return nums[0]
	}
	if len(nums) == 2 {
		return max(nums[0], nums[1], nums[0]+nums[1])
	}
	dp := make([]int, len(nums))
	dp[0] = nums[0]
	ans := dp[0]
	sum := nums[0]
	for i := 1; i < len(nums); i++ {
		sum += nums[i]
		dp[i] = max(dp[i-1]+nums[i], nums[i])
		ans = max(ans, dp[i])
	}
	//第二种情况 最大和跨越了头尾，即在中间找一个最小和
	var minSubarraySum func(nums []int) int
	minSubarraySum = func(nums []int) int {
		dp := make([]int, len(nums))
		dp[0] = nums[0]
		ans := dp[0]
		for i := 1; i < len(nums); i++ {
			dp[i] = min(dp[i-1]+nums[i], nums[i])
			ans = min(ans, dp[i])
		}
		return ans
	}
	minSum := minSubarraySum(nums[1 : len(nums)-1])
	ans = max(ans, sum-minSum)
	return ans
}

// 2560. 打家劫舍 IV 二分答案法
func minCapability(nums []int, k int) int {
	//能力越大 能偷的房子越多 越能满足至少偷k间的要求
	var check func(ability int) bool
	check = func(ability int) bool {
		if len(nums) == 1 {
			return ability >= nums[0]
		}
		if len(nums) == 2 {
			return ability >= nums[0] || ability >= nums[1]
		}
		//dpi 到i号房间为止 能偷的最大房间个数
		dp := make([]int, len(nums))
		if ability >= nums[0] {
			dp[0] = 1
			dp[1] = 1
		} else {
			dp[0] = 0
			if ability >= nums[1] {
				dp[1] = 1
			}
		}
		for i := 2; i < len(nums); i++ {
			if ability >= nums[i] {
				dp[i] = max(dp[i-2]+1, dp[i-1])
			} else { //偷不了
				dp[i] = dp[i-1]
			}
		}
		return dp[len(nums)-1] >= k
	}
	maxAbility := slices.Max(nums)
	left, right := 1, maxAbility
	var ans int
	for left <= right {
		mid := (right-left)/2 + left
		if check(mid) {
			right = mid - 1
			ans = mid
		} else {
			left = mid + 1
		}
	}
	return ans
}

// 239. 滑动窗口最大值 单调队列
func maxSlidingWindow(nums []int, k int) (ans []int) {
	//维护单调递减的单调队列
	maxQueue := make([]int, 0, len(nums))
	for i := 0; i < k-1; i++ {
		for len(maxQueue) > 0 && nums[maxQueue[len(maxQueue)-1]] <= nums[i] {
			maxQueue = maxQueue[:len(maxQueue)-1]
		}
		maxQueue = append(maxQueue, i)
	}
	for i := 0; i <= len(nums)-k; i++ {
		//加入一个到队列 收集答案 再弹出一个
		for len(maxQueue) > 0 && nums[maxQueue[len(maxQueue)-1]] <= nums[i+k-1] {
			maxQueue = maxQueue[:len(maxQueue)-1]
		}
		maxQueue = append(maxQueue, i+k-1)
		//收集当前最大值
		ans = append(ans, nums[maxQueue[0]])
		if maxQueue[0] == i {
			maxQueue = maxQueue[1:]
		}
	}
	return
}

//  82. 删除排序链表中的重复元素 II 迭代法
//     func deleteDuplicates(head *ListNode) *ListNode {
//     dummy := &ListNode{Next: head}
//     slow, fast := dummy, head
//     for fast != nil && fast.Next != nil {
//     if fast.Val == fast.Next.Val {
//     dup := fast.Val
//     for fast != nil && fast.Val == dup {
//     fast = fast.Next
//     }
//     } else {
//     slow.Next.Val = fast.Val
//     slow = slow.Next
//     fast = fast.Next
//     }
//     }
//     if fast != nil {
//     slow.Next.Val = fast.Val
//     slow = slow.Next
//     }
//     slow.Next = nil
//     return dummy.Next
//     }
//
// 82. 删除排序链表中的重复元素 II 递归法
//func deleteDuplicates(head *ListNode) *ListNode {
//	var dfs func(node *ListNode) *ListNode
//	dfs = func(node *ListNode) *ListNode {
//		if node == nil || node.Next == nil {
//			return node
//		}
//		if node.Val != node.Next.Val {
//			node.Next = dfs(node.Next)
//			return node
//		} else {
//			val := node.Val
//			for node != nil && node.Val == val {
//				node = node.Next
//			}
//			return dfs(node)
//		}
//	}
//	return dfs(head)
//}
