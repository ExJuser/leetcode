package main

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strconv"
)

type ListNode struct {
	Val  int
	Next *ListNode
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// ————————————————————————————————————————————————————————————————————————————————
// 3. 无重复字符的最长子串
// 使用哈希表的做法
func lengthOfLongestSubstring(s string) int {
	n := len(s)
	mp := make(map[byte]int, n)
	var left, ans int
	for right := 0; right < n; right++ {
		mp[s[right]]++
		//没有重复字符的话二者应该相等
		for ; right-left+1 > len(mp); left++ {
			mp[s[left]]--
			if mp[s[left]] == 0 {
				delete(mp, s[left])
			}
		}
		//从循环中退出代表没有重复字符 此时收集最大答案
		ans = max(ans, right-left+1)
	}
	return ans
}

// 使用数组的做法 空间复杂度更好一些
func lengthOfLongestSubstring2(s string) int {
	mp := make([]int, math.MaxUint8)
	n := len(s)
	var left, ans int
	for right := 0; right < n; right++ {
		mp[s[right]]++
		//意味着无重复字符
		if mp[s[right]] == 1 {
			ans = max(ans, right-left+1)
		} else { //滑动窗口就需要移动到直到重复字符不再重复
			for ; mp[s[right]] > 1; left++ {
				mp[s[left]]--
			}
		}
	}
	return ans
}

// ————————————————————————————————————————————————————————————————————————————————
// 206. 反转链表
// 迭代法完成
func reverseList(head *ListNode) *ListNode {
	var pre *ListNode
	for cur := head; cur != nil; {
		nxt := cur.Next
		cur.Next = pre
		cur, pre = nxt, cur
	}
	return pre
}

// 递归法完成
func reverseList2(head *ListNode) *ListNode {
	var dfs func(node *ListNode) *ListNode
	dfs = func(node *ListNode) *ListNode {
		if node == nil || node.Next == nil {
			return node
		}
		newHead := dfs(node.Next)
		node.Next.Next = node
		node.Next = nil
		return newHead
	}
	return dfs(head)
}

// ————————————————————————————————————————————————————————————————————————————————

// LRUCache 146. LRU 缓存  0816 BUGFree 一遍过
type LRUCache struct {
	list      *DoubleList
	keyToNode map[int]*DoubleNode
	capacity  int
}

func Constructor(capacity int) LRUCache {
	return LRUCache{
		list:      NewDoubleList(),
		keyToNode: make(map[int]*DoubleNode),
		capacity:  capacity,
	}
}

func (this *LRUCache) Get(key int) int {
	//先从keyToNode中找到对应的节点
	if node, ok := this.keyToNode[key]; ok { //如果存在 返回并将其移动到队首
		this.list.MoveToFront(node)
		return node.Val
	}
	return -1
}

func (this *LRUCache) Put(key int, value int) {
	//如果keyToNode存在 修改他的值 移动到队首
	if node, ok := this.keyToNode[key]; ok {
		node.Val = value
		this.list.MoveToFront(node)
	} else { //不存在需要插入
		newNode := &DoubleNode{
			Key: key,
			Val: value,
		}
		this.list.PushFront(newNode)
		this.keyToNode[key] = newNode
		//如果超出容量 需要删除最久未使用的 即list.back
		if len(this.keyToNode) > this.capacity {
			back := this.list.Back()
			this.list.Remove(back)
			delete(this.keyToNode, back.Key)
		}
	}
}

type DoubleNode struct {
	Key  int
	Val  int
	Next *DoubleNode
	Prev *DoubleNode
	//如果要实现过期删除 可以把过期时间设置在这里
	//然后启动一个定期的异步协程 清理 并且每次访问也都检查一下是否过期
}

type DoubleList struct {
	DummyHead *DoubleNode
	DummyTail *DoubleNode
}

func NewDoubleList() *DoubleList {
	dummyHead := &DoubleNode{}
	dummyTail := &DoubleNode{}
	dummyHead.Next = dummyTail
	dummyTail.Prev = dummyHead
	return &DoubleList{
		DummyHead: dummyHead,
		DummyTail: dummyTail,
	}
}

func (d *DoubleList) MoveToFront(node *DoubleNode) {
	//将node移动到队首 需要现将其从原位置删除 再插入到头部
	d.Remove(node)
	d.PushFront(node)
}

func (d *DoubleList) Remove(node *DoubleNode) {
	node.Prev.Next = node.Next
	node.Next.Prev = node.Prev
	node.Prev = nil
	node.Next = nil
}

func (d *DoubleList) PushFront(node *DoubleNode) {
	node.Next = d.DummyHead.Next
	node.Next.Prev = node
	node.Prev = d.DummyHead
	d.DummyHead.Next = node
}
func (d *DoubleList) Back() *DoubleNode {
	return d.DummyTail.Prev
}

// ——————————————————————————————————————————————————————————————————————————
// 215. 数组中的第K个最大元素

type minHeap []int

func (m *minHeap) Len() int {
	return len(*m)
}

func (m *minHeap) Less(i, j int) bool {
	return (*m)[i] < (*m)[j]
}

func (m *minHeap) Swap(i, j int) {
	(*m)[i], (*m)[j] = (*m)[j], (*m)[i]
}

func (m *minHeap) Push(x any) {
	*m = append(*m, x.(int))
}

func (m *minHeap) Pop() any {
	x := (*m)[m.Len()-1]
	*m = (*m)[:m.Len()-1]
	return x
}

// 堆排序实现
//func findKthLargest(nums []int, k int) int {
//	hp := &minHeap{}
//	for _, num := range nums {
//		heap.Push(hp, num)
//		if hp.Len() > k {
//			heap.Pop(hp)
//		}
//	}
//	return (*hp)[0]
//}

// 快速选择实现
//func findKthLargest2(nums []int, k int) int {
//	var helper func(left, right, k int) int
//	helper = func(left, right, k int) int {
//		// MARK 第一个注意点
//		if left >= right {
//			return nums[k]
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
//		// MARK 第二个注意点
//		if j >= k {
//			return helper(left, j, k)
//		} else {
//			return helper(i, right, k)
//		}
//	}
//	return helper(0, len(nums)-1, len(nums)-k)
//}

// ——————————————————————————————————————————————————————————
// 25. K 个一组翻转链表

func getListLength(head *ListNode) int {
	var ans int
	for cur := head; cur != nil; cur = cur.Next {
		ans++
	}
	return ans
}

//func reverseKGroup(head *ListNode, k int) *ListNode {
//	//有可能修改队首节点的都创建一个虚拟头节点方便边界的处理
//	dummy := &ListNode{Next: head}
//	//首先需要得到链表的长度
//	n := getListLength(head)
//	var temp = dummy
//	var cur = head
//	var pre *ListNode
//	for i := 0; i < n/k; i++ {
//		for j := 0; j < k; j++ {
//			nxt := cur.Next
//			cur.Next = pre
//			pre, cur = cur, nxt
//		}
//		newTemp := temp.Next
//		temp.Next.Next = cur
//		temp.Next = pre
//		temp = newTemp
//	}
//	return dummy.Next
//}

// 递归解法
func reverseKGroup2(head *ListNode, k int) *ListNode {
	var dfs func(node *ListNode) *ListNode
	dfs = func(node *ListNode) *ListNode {
		//如果剩下的节点不足k个直接返回
		if getListLength(node) < k {
			return node
		}
		var pre *ListNode
		cur := node
		for i := 0; i < k; i++ {
			nxt := cur.Next
			cur.Next = pre
			cur, pre = nxt, cur
		}
		node.Next = dfs(cur)
		return pre
	}
	return dfs(head)
}

// ——————————————————————————————————————————————————————————
// 15. 三数之和
// 熟记并背诵
//func threeSum(nums []int) (ans [][]int) {
//	slices.Sort(nums)
//	for i := 0; i < len(nums); i++ {
//		n1 := nums[i]
//		if n1 <= 0 && (i == 0 || n1 != nums[i-1]) {
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

// ————————————————————————————————————————————————————————————————
// 53. 最大子数组和 dp版本
func maxSubArray(nums []int) int {
	//dpi 以numsi为最后一个元素的最大子数组和
	dp := make([]int, len(nums))
	dp[0] = nums[0]
	var ans = dp[0]
	for i := 1; i < len(nums); i++ {
		dp[i] = max(dp[i-1]+nums[i], nums[i])
		ans = max(ans, dp[i])
	}
	return ans
}

func maxSubArray2(nums []int) int {
	//dpi 以numsi为最后一个元素的最大子数组和
	//dp := make([]int, len(nums))
	var dp0, dp1 int
	var ans = math.MinInt
	for i := 0; i < len(nums); i++ {
		dp1 = max(dp0+nums[i], nums[i])
		ans = max(ans, dp1)
		dp0 = dp1
	}
	return ans
}

//————————————————————————————————————————————————————

// 手撕快速排序
func sortArray(nums []int) []int {
	var helper func(left, right int)
	helper = func(left, right int) {
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
		helper(left, j)
		helper(i, right)
	}
	helper(0, len(nums)-1)
	return nums
}

// ————————————————————————————————————————————————————————
// 21. 合并两个有序链表
func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	dummy := &ListNode{}
	cur := dummy
	p1, p2 := list1, list2
	for p1 != nil && p2 != nil {
		if p1.Val <= p2.Val {
			cur.Next = &ListNode{Val: p1.Val}
			p1 = p1.Next
		} else {
			cur.Next = &ListNode{Val: p2.Val}
			p2 = p2.Next
		}
		cur = cur.Next
	}
	if p1 != nil {
		cur.Next = p1
	} else {
		cur.Next = p2
	}
	return dummy.Next
}

// 使用递归实现的版本
func mergeTwoLists2(list1, list2 *ListNode) *ListNode {
	if list1 == nil {
		return list2 // 注：如果都为空则返回空
	}
	if list2 == nil {
		return list1
	}
	if list1.Val < list2.Val {
		list1.Next = mergeTwoLists(list1.Next, list2)
		return list1
	}
	list2.Next = mergeTwoLists(list1, list2.Next)
	return list2
}

// 带去重的版本
func mergeTwoListsDuplicate(list1 *ListNode, list2 *ListNode) *ListNode {
	dummy := &ListNode{Val: math.MinInt}
	cur := dummy
	p1, p2 := list1, list2
	for p1 != nil || p2 != nil {
		var val int
		//如果都不为空 正常判断大小
		if p1 != nil && p2 != nil {
			if p1.Val <= p2.Val {
				val = p1.Val
				p1 = p1.Next
			} else {
				val = p2.Val
				p2 = p2.Next
			}
		} else if p1 != nil {
			val = p1.Val
			p1 = p1.Next
		} else {
			val = p2.Val
			p2 = p2.Next
		}
		if cur.Val == math.MinInt || cur.Val != val {
			cur.Next = &ListNode{Val: val}
			cur = cur.Next
		}
	}
	return dummy.Next
}

// ——————————————————————————————————————————————————————
// 1. 两数之和
func twoSum(nums []int, target int) []int {
	mp := make(map[int]int, len(nums))
	for i := 0; i < len(nums); i++ {
		if j, ok := mp[target-nums[i]]; ok {
			return []int{i, j}
		}
		mp[nums[i]] = i
	}
	return []int{}
}

// ————————————————————————————————————————————————————————————————
// 5. 最长回文子串
func longestPalindrome(s string) string {
	n := len(s)
	dp := make([][]bool, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]bool, n)
	}
	var ans string
	for i := n - 1; i >= 0; i-- {
		for j := i; j < n; j++ {
			if i == j {
				dp[i][j] = true
			} else if j-i == 1 {
				dp[i][j] = s[i] == s[j]
			} else {
				if s[i] == s[j] {
					dp[i][j] = dp[i+1][j-1]
				}
			}
			//每次都在这里错 首先得是回文子串才行
			if dp[i][j] && j-i+1 > len(ans) {
				ans = s[i : j+1]
			}
		}
	}
	return ans
}

// 102. 二叉树的层序遍历
//func levelOrder(root *TreeNode) (ans [][]int) {
//	if root == nil {
//		return
//	}
//	queue := make([]*TreeNode, 0, 2000)
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

// 层序遍历的递归实现
func levelOrder2(root *TreeNode) (ans [][]int) {
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

// 88. 合并两个有序数组 倒序
// MARK 还不熟练
//func merge(nums1 []int, m int, nums2 []int, n int) {
//	i, j := m-1, n-1
//	insertIndex := m + n - 1
//	for j >= 0 {
//		if i >= 0 && nums1[i] >= nums2[j] {
//			nums1[insertIndex] = nums1[i]
//			i--
//		} else {
//			nums1[insertIndex] = nums2[j]
//			j--
//		}
//		insertIndex--
//	}
//}

// 27. 移除元素
func removeElement(nums []int, val int) int {
	var index int
	for i := 0; i < len(nums); i++ {
		if nums[i] != val {
			nums[index] = nums[i]
			index++
		}
	}
	return index
}

// 26. 删除有序数组中的重复项
func removeDuplicates(nums []int) int {
	//遍历到一个数字 填入index 向后遍历到第一个不重复的数字
	var index int
	for i := 0; i < len(nums); {
		nums[index] = nums[i]
		for i < len(nums) && nums[i] == nums[index] {
			i++
		}
		index++
	}
	return index
}

// 80. 删除有序数组中的重复项 II
// 可以重复两次 mark 还不熟练
func removeDuplicates2(nums []int) int {
	//遍历到一个数 记录一个计数器 如果和上一个数相等 计数器++
	//如果计数器为2 就向后遍历到第一个不相等的位置 清空计数器
	var index int
	var counter int
	for i := 0; i < len(nums); {
		if index > 0 && nums[i] == nums[index-1] {
			counter++
		} else {
			counter = 1
		}
		if counter > 2 {
			for i < len(nums) && nums[i] == nums[index-1] {
				i++
			}
		} else {
			nums[index] = nums[i]
			index++
			i++
		}
	}
	return index
}

// 169. 多数元素 快速选择算法 一定是中间值
func majorityElement(nums []int) int {
	var helper func(left, right, k int) int
	helper = func(left, right, k int) int {
		if left >= right {
			return nums[k]
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
		if j >= k {
			return helper(left, j, k)
		} else {
			return helper(i, right, k)
		}
	}
	return helper(0, len(nums)-1, len(nums)/2)
}

// 189. 轮转数组 mark 轮转范围注意
//func rotate(nums []int, k int) {
//	k %= len(nums)
//	slices.Reverse(nums[:len(nums)-k])
//	slices.Reverse(nums[len(nums)-k:])
//	slices.Reverse(nums)
//}

// 121. 买卖股票的最佳时机 动态规划解法
func maxProfit(prices []int) int {
	dp := make([][2]int, len(prices))
	dp[0][0] = -prices[0]
	for i := 1; i < len(prices); i++ {
		//持有股票 要么前一天就持有 要么之前从来没有持有过 第一次购买
		dp[i][0] = max(dp[i-1][0], -prices[i])
		//不持有股票 要么之前持有 要么今天卖出
		dp[i][1] = max(dp[i-1][1], dp[i-1][0]+prices[i])
	}
	return dp[len(prices)-1][1]
}

// 最优解 后面的最大的减去前面的最小的
//
//	func maxProfit2(prices []int) int {
//		var ans int
//		lowestPrice := prices[0]
//		for i := 1; i < len(prices); i++ {
//			lowestPrice = min(lowestPrice, prices[i])
//			ans = max(ans, prices[i]-lowestPrice)
//		}
//		return ans
//	}
//
// 122. 买卖股票的最佳时机 II
func maxProfit2(prices []int) int {
	dp := make([][2]int, len(prices))
	dp[0][0] = -prices[0]
	for i := 1; i < len(prices); i++ {
		dp[i][0] = max(dp[i-1][0], dp[i-1][1]-prices[i])
		dp[i][1] = max(dp[i-1][1], dp[i-1][0]+prices[i])
	}
	return dp[len(prices)-1][1]
}

// 55. 跳跃游戏
func canJump(nums []int) bool {
	maxRight := 0
	for i := 0; i < len(nums) && i <= maxRight; i++ {
		maxRight = max(maxRight, nums[i]+i)
	}
	return maxRight >= len(nums)-1
}

// 45. 跳跃游戏 II
func jump(nums []int) int {
	//dpi跳跃到i位置的最小跳跃次数
	dp := make([]int, len(nums))
	for i := 1; i < len(nums); i++ {
		dp[i] = math.MaxInt
	}
	for i := 0; i < len(nums); i++ {
		for j := 1; j <= nums[i] && j < len(nums)-i; j++ {
			dp[i+j] = min(dp[i+j], dp[i]+1)
		}
	}
	return dp[len(nums)-1]
}

// 33. 搜索旋转排序数组 互不相同 画图即可
func search(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := (right-left)/2 + left
		if nums[mid] == target {
			return mid
		}
		if nums[mid] >= nums[left] {
			if target >= nums[left] && target < nums[mid] {
				right = mid - 1
			} else {
				left = mid + 1
			}
		} else {
			if target <= nums[right] && target > nums[mid] {
				left = mid + 1
			} else {
				right = mid - 1
			}
		}
	}
	return -1
}

// 200. 岛屿数量 dfs
//func numIslands(grid [][]byte) int {
//	directions := [][]int{
//		{0, 1}, {0, -1}, {1, 0}, {-1, 0},
//	}
//	var dfs func(i, j int)
//	dfs = func(i, j int) {
//		//位置不合法或者访问过或者不是陆地
//		if i < 0 || i >= len(grid) || j < 0 || j >= len(grid[0]) || grid[i][j] != '1' {
//			return
//		}
//		grid[i][j] = '2'
//		for _, d := range directions {
//			ii, jj := i+d[0], j+d[1]
//			dfs(ii, jj)
//		}
//	}
//	var cnt int
//	for i := 0; i < len(grid); i++ {
//		for j := 0; j < len(grid[0]); j++ {
//			if grid[i][j] == '1' {
//				cnt++
//				dfs(i, j)
//			}
//		}
//	}
//	return cnt
//}

// 200. 岛屿数量 bfs
func numIslands2(grid [][]byte) int {
	directions := [][]int{
		{0, 1}, {0, -1}, {1, 0}, {-1, 0},
	}
	var bfs func(i, j int)
	bfs = func(i, j int) {
		queue := make([][2]int, 0)
		queue = append(queue, [2]int{i, j})
		for len(queue) > 0 {
			x := queue[0]
			queue = queue[1:]
			for _, d := range directions {
				ii, jj := x[0]+d[0], x[1]+d[1]
				if ii >= 0 && ii < len(grid) && jj >= 0 && jj < len(grid[0]) && grid[ii][jj] == '1' {
					grid[ii][jj] = '2'
					queue = append(queue, [2]int{ii, jj})
				}
			}
		}
	}
	var cnt int
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == '1' {
				cnt++
				grid[i][j] = '2'
				bfs(i, j)
			}
		}
	}
	return cnt
}

// 695. 岛屿的最大面积
//func maxAreaOfIsland(grid [][]int) int {
//	directions := [][]int{
//		{0, 1}, {0, -1}, {1, 0}, {-1, 0},
//	}
//	var dfs func(i, j int) int
//	dfs = func(i, j int) int {
//		//位置不合法或者访问过或者不是陆地
//		if i < 0 || i >= len(grid) || j < 0 || j >= len(grid[0]) || grid[i][j] != 1 {
//			return 0
//		}
//		grid[i][j] = 2
//		cnt := 1
//		for _, d := range directions {
//			ii, jj := i+d[0], j+d[1]
//			cnt += dfs(ii, jj)
//		}
//		return cnt
//	}
//	var ans int
//	for i := 0; i < len(grid); i++ {
//		for j := 0; j < len(grid[0]); j++ {
//			if grid[i][j] == 1 {
//				ans = max(ans, dfs(i, j))
//			}
//		}
//	}
//	return ans
//}

// LC827
// 463. 岛屿的周长
func islandPerimeter(grid [][]int) int {
	//对于每一个为1的方块：
	//如果四周为边界或者为0 计算一个周长
	directions := [][]int{
		{0, 1}, {0, -1}, {1, 0}, {-1, 0},
	}
	var ans int
	var computePerimeter func(i, j int)
	computePerimeter = func(i, j int) {
		for _, d := range directions {
			ii, jj := i+d[0], j+d[1]
			if ii < 0 || ii >= len(grid) || jj < 0 || jj >= len(grid[0]) || grid[ii][jj] == 0 {
				ans += 1
			}
		}
	}
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == 1 {
				computePerimeter(i, j)
			}
		}
	}
	return ans
}

// 20. 有效的括号
func isValid(s string) bool {
	mapping := map[byte]byte{
		')': '(',
		']': '[',
		'}': '{',
	}
	stack := make([]byte, 0, len(s))
	for _, ch := range s {
		if ch == '(' || ch == '[' || ch == '{' {
			stack = append(stack, byte(ch))
		} else {
			if len(stack) == 0 || mapping[byte(ch)] != stack[len(stack)-1] {
				return false
			}
			stack = stack[:len(stack)-1]
		}
	}
	return len(stack) == 0
}

// 46. 全排列
func permute(nums []int) (ans [][]int) {
	visited := make([]bool, len(nums))
	var dfs func(path []int)
	dfs = func(path []int) {
		if len(path) == len(nums) {
			ans = append(ans, append([]int{}, path...))
			return
		}
		for i := 0; i < len(nums); i++ {
			if !visited[i] {
				visited[i] = true
				path = append(path, nums[i])
				dfs(path)
				path = path[:len(path)-1]
				visited[i] = false
			}
		}
	}
	dfs([]int{})
	return
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

// 求一个数的平方根，保留两位小数，不能用api
func sqrt(num int) float64 {
	//先求整数部分
	left, right := 0, num
	var res int
	for left <= right {
		mid := (right-left)/2 + left
		if mid*mid <= num {
			res = mid
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	if res*res == num {
		return float64(res)
	}
	//含有小数部分
	var resFloat float64
	target := float64(num)
	var leftFloat, rightFloat = float64(res) + 0.01, float64(res) + 0.99
	for leftFloat <= rightFloat {
		mid, _ := strconv.ParseFloat(fmt.Sprintf("%.2f", (rightFloat-leftFloat)/2+leftFloat), 64)
		if mid*mid <= target {
			resFloat = mid
			leftFloat = mid + 0.01
		} else {
			rightFloat = mid - 0.01
		}
	}
	return resFloat
}

// 48. 旋转图像 顺时针旋转
func rotate(matrix [][]int) {
	for i := 0; i < len(matrix); i++ {
		for j := i + 1; j < len(matrix[i]); j++ {
			matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
		}
	}
	for _, row := range matrix {
		slices.Reverse(row)
	}
}

// 160. 相交链表 哈希表法
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	set := make(map[*ListNode]struct{})
	for cur := headA; cur != nil; cur = cur.Next {
		set[cur] = struct{}{}
	}
	for cur := headB; cur != nil; cur = cur.Next {
		if _, ok := set[cur]; ok {
			return cur
		}
	}
	return nil
}

func getIntersectionNode2(headA, headB *ListNode) *ListNode {
	//分别求出两段链表的长度
	lengthA := getListLength(headA)
	lengthB := getListLength(headB)
	diff := lengthA - lengthB
	curA, curB := headA, headB
	if diff > 0 { //a先走
		for i := 0; i < diff; i++ {
			curA = curA.Next
		}
	} else { //b先走
		for i := 0; i < (-diff); i++ {
			curB = curB.Next
		}
	}
	//两个同步走
	for curA != nil && curB != nil {
		if curA == curB {
			return curA //交点
		}
		curA = curA.Next
		curB = curB.Next
	}
	return nil
}

// 234. 回文链表
//func isPalindrome(head *ListNode) bool {
//	//找到链表中点然后翻转 再逐一比较
//	slow, fast := head, head
//	for fast != nil && fast.Next != nil {
//		slow = slow.Next
//		fast = fast.Next.Next
//	}
//	reversed := reverseList(slow)
//	for p1, p2 := head, reversed; p1 != nil && p2 != nil; p1, p2 = p1.Next, p2.Next {
//		if p1.Val != p2.Val {
//			return false
//		}
//	}
//	return true
//}

// 142. 环形链表 II
func detectCycle(head *ListNode) *ListNode {
	//先确定是否有环
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast { //有环
			cur := head
			for cur != slow {
				cur = cur.Next
				slow = slow.Next
			}
			return slow
		}
	}
	return nil
}

// 94. 二叉树的中序遍历
func inorderTraversal(root *TreeNode) (ans []int) {
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		ans = append(ans, node.Val)
		dfs(node.Right)
	}
	dfs(root)
	return
}

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

// 543. 二叉树的直径
func diameterOfBinaryTree(root *TreeNode) int {
	var ans int
	var dfs func(node *TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		//经过一个节点的直径：左边的最长路径+右边的最长路径
		left, right := dfs(node.Left), dfs(node.Right)
		ans = max(ans, left+right)
		return max(left, right) + 1
	}
	dfs(root)
	return ans
}

// 226. 翻转二叉树 递归
func invertTree(root *TreeNode) *TreeNode {
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

// 226. 翻转二叉树 迭代
func invertTree2(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	stack := make([]*TreeNode, 0)
	stack = append(stack, root)
	for len(stack) > 0 {
		x := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		x.Left, x.Right = x.Right, x.Left
		if x.Right != nil {
			stack = append(stack, x.Right)
		}
		if x.Left != nil {
			stack = append(stack, x.Left)
		}
	}
	return root
}

//	func calculate(s string) int {
//		//如果是数字 检查它左边是不是数字 如果是 说明数字不是一位数 和numstack里面的数字做运算
//		//如果是数字 左边不是数字 检查numstack栈顶有没有数字 如果有 取出一个运算符做计算 再入栈
//		//如果栈顶不是数字 直接入numstack栈
//
//		//如果是符号、左括号 直接入栈
//		//如果是右括号 遍历到上一个右括号 取出两侧的括号
//		//如果是空格 忽略
//		numStack := make([]int, 0, len(s))
//		opStack := make([]byte, 0, len(s))
//		for i := 0; i < len(s); {
//			if s[i] == ' ' || s[i] == '(' || s[i] == ')' { //忽略空格
//				i++
//			} else if unicode.IsDigit(rune(s[i])) { //遇到数字应该向后遍历到数字结束 收集这个数字
//				j := i + 1
//				for j < len(s) && unicode.IsDigit(rune(s[j])) {
//					j++
//				}
//				num, _ := strconv.Atoi(s[i:j])
//				numStack = append(numStack, num)
//				//如果此时数字栈顶不为空 取出一个操作符计算
//				i = j
//
//				if len(opStack)+1 != len(numStack) {
//					//前一个符号是单目运算的负号
//					numStack[len(numStack)-1] *= -1
//					opStack = opStack[:len(opStack)-1]
//				}
//
//				for len(numStack) > 0 && len(opStack) > 0 {
//					num1 := numStack[len(numStack)-1]
//					numStack = numStack[:len(numStack)-1]
//					num2 := numStack[len(numStack)-1]
//					numStack = numStack[:len(numStack)-1]
//					op := opStack[len(opStack)-1]
//					opStack = opStack[:len(opStack)-1]
//					var res int
//					if op == '+' {
//						res = num2 + num1
//					} else {
//						res = num2 - num1
//					}
//					numStack = append(numStack, res)
//				}
//			} else if s[i] == '+' || s[i] == '-' { //如果是操作符
//				opStack = append(opStack, s[i])
//				i++
//			}
//		}
//		return numStack[0]
//	}
//
// 54. 螺旋矩阵 mark 边界条件
func spiralOrder(matrix [][]int) (ans []int) {
	top, bottom := 0, len(matrix)-1
	left, right := 0, len(matrix[0])-1
	for left < right && top < bottom {
		for i := left; i <= right; i++ {
			ans = append(ans, matrix[top][i])
		}
		top++
		for i := top; i <= bottom; i++ {
			ans = append(ans, matrix[i][right])
		}
		right--
		for i := right; i >= left; i-- {
			ans = append(ans, matrix[bottom][i])
		}
		bottom--
		for i := bottom; i >= top; i-- {
			ans = append(ans, matrix[i][left])
		}
		left++
	}
	if left == right {
		for i := top; i <= bottom; i++ {
			ans = append(ans, matrix[i][left])
		}
	} else if top == bottom {
		for i := left; i <= right; i++ {
			ans = append(ans, matrix[top][i])
		}
	}
	return
}

// 2. 两数相加
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	dummy := &ListNode{}
	var carry int
	cur := dummy
	for l1 != nil || l2 != nil || carry != 0 {
		cur.Next = &ListNode{}
		cur = cur.Next
		if l1 != nil {
			cur.Val += l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			cur.Val += l2.Val
			l2 = l2.Next
		}
		if carry != 0 {
			cur.Val += carry
		}
		carry = cur.Val / 10
		cur.Val %= 10
	}
	return dummy.Next
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

// 23. 合并 K 个升序链表
func mergeKLists(lists []*ListNode) *ListNode {
	hp := &NodeHeap{}
	for _, list := range lists {
		if list != nil {
			heap.Push(hp, list)
		}
	}
	dummy := &ListNode{}
	cur := dummy
	for hp.Len() > 0 {
		x := heap.Pop(hp).(*ListNode)
		cur.Next = &ListNode{Val: x.Val}
		cur = cur.Next
		if x.Next != nil {
			heap.Push(hp, x.Next)
		}
	}
	return dummy.Next
}

// 56. 合并区间
//func merge(intervals [][]int) (ans [][]int) {
//	slices.SortFunc(intervals, func(a, b []int) int {
//		return a[0] - b[0]
//	})
//	pre := intervals[0]
//	for i := 1; i < len(intervals); i++ {
//		cur := intervals[i]
//		if cur[0] > pre[1] {
//			ans = append(ans, pre)
//			pre = cur
//		} else {
//			pre[1] = max(pre[1], cur[1])
//		}
//	}
//	ans = append(ans, pre)
//	return
//}

// 19. 删除链表的倒数第 N 个结点
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummy := &ListNode{Next: head}
	slow, fast := dummy, dummy
	for ; n > 0; n-- {
		fast = fast.Next
	}
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next
	}
	slow.Next = slow.Next.Next
	return dummy.Next
}

func isSameTree(p, q *TreeNode) bool {
	if p == nil || q == nil {
		return p == q
	}
	return p.Val == q.Val && isSameTree(p.Left, q.Right) && isSameTree(p.Right, q.Left)
}

func isSymmetric(root *TreeNode) bool {
	return isSameTree(root.Left, root.Right)
}

// 136. 只出现一次的数字
// 相同的数字异或结果为0：出现两次为0
// 任何数异或0等于自身
// 异或满足交换律
func singleNumber(nums []int) int {
	var res int
	for _, num := range nums {
		res ^= num
	}
	return res
}

// 236. 二叉树的最近公共祖先
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	if root == p || root == q {
		return root
	}
	left := lowestCommonAncestor(root.Left, p, q)
	right := lowestCommonAncestor(root.Right, p, q)
	if left != nil && right != nil {
		return root
	}
	if left == nil {
		return right
	}
	return left
}

// 98. 验证二叉搜索树 中序序列为有序序列
func isValidBST(root *TreeNode) bool {
	sequence := make([]int, 0)
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		sequence = append(sequence, node.Val)
		dfs(node.Right)
	}
	dfs(root)
	for i := 0; i < len(sequence)-1; i++ {
		if sequence[i] >= sequence[i+1] {
			return false
		}
	}
	return true
}

// 78. 子集
func subsets(nums []int) (ans [][]int) {
	var dfs func(index int, path []int)
	dfs = func(index int, path []int) {
		if index == len(nums) {
			ans = append(ans, append([]int{}, path...))
			return
		}
		path = append(path, nums[index])
		dfs(index+1, path)
		path = path[:len(path)-1]
		dfs(index+1, path)
	}
	dfs(0, []int{})
	return
}

// 39. 组合总和 数字可以被无限选取
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
		path = append(path, candidates[index])
		dfs(index, sum+candidates[index], path)
		path = path[:len(path)-1]
		dfs(index+1, sum, path)
	}
	dfs(0, 0, []int{})
	return
}

func mergeList(l1, l2 *ListNode) *ListNode {
	dummy := &ListNode{}
	cur := dummy
	for l1 != nil && l2 != nil {
		cur.Next = &ListNode{}
		cur = cur.Next
		if l1.Val <= l2.Val {
			cur.Val = l1.Val
			l1 = l1.Next
		} else {
			cur.Val = l2.Val
			l2 = l2.Next
		}
	}
	if l1 == nil {
		cur.Next = l2
	} else {
		cur.Next = l1
	}
	return dummy.Next
}

// 148. 排序链表 归并排序
func sortList(head *ListNode) *ListNode {
	var dfs func(node *ListNode) *ListNode
	dfs = func(node *ListNode) *ListNode {
		if node == nil || node.Next == nil {
			return node
		}
		var preSlow *ListNode
		slow, fast := node, node
		for fast != nil && fast.Next != nil {
			preSlow = slow
			slow = slow.Next
			fast = fast.Next.Next
		}
		preSlow.Next = nil
		return mergeList(dfs(node), dfs(slow))
	}
	return dfs(head)
}

// 147. 对链表进行插入排序
func insertionSortList(head *ListNode) *ListNode {
	dummy := &ListNode{Next: head}
	pre := dummy
	for cur := head; cur != nil; {
		nxt := cur.Next
		if pre != dummy && cur.Val < pre.Val {
			pre.Next = nxt
			cur.Next = nil
			i := dummy
			for i.Next != nil && cur.Val >= i.Next.Val {
				i = i.Next
			}
			cur.Next = i.Next
			i.Next = cur
		} else {
			pre = cur
		}
		cur = nxt
	}
	return dummy.Next
}
