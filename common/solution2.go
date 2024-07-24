package common

import (
	"container/heap"
	"math/rand/v2"
	"slices"
	"sort"
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
func numIslands(grid [][]byte) (ans int) {
	var bfs func(i, j int)
	bfs = func(i, j int) {
		if i < 0 || j < 0 || i >= len(grid) || j >= len(grid[0]) {
			return
		}
		if grid[i][j] == '1' {
			grid[i][j] = '0'
			queue := [][2]int{{i, j}}
			for len(queue) > 0 {
				x, y := queue[0][0], queue[0][1]
				queue = queue[1:]
				bfs(x-1, y)
				bfs(x+1, y)
				bfs(x, y-1)
				bfs(x, y+1)
			}
		}
	}
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == '1' {
				ans++
				bfs(i, j)
			}
		}
	}
	return ans
}

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

func zigzagLevelOrder(root *TreeNode) [][]int {
	levels := levelOrder(root)
	for i := 0; i < len(levels); i++ {
		if i%2 == 1 {
			slices.Reverse(levels[i])
		}
	}
	return levels
}

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
