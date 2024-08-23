package main

import (
	"container/heap"
	"math"
	"slices"
)

// 49. 字母异位词分组
func groupAnagrams(strs []string) (ans [][]string) {
	mp := make(map[string]int)
	for _, str := range strs {
		bytes := []byte(str)
		slices.Sort(bytes)
		sorted := string(bytes)
		if i, ok := mp[sorted]; ok {
			ans[i] = append(ans[i], str)
		} else {
			mp[sorted] = len(ans)
			ans = append(ans, []string{str})
		}
	}
	return
}

// 128. 最长连续序列
func longestConsecutive(nums []int) int {
	mp := make(map[int]struct{})
	for _, num := range nums {
		mp[num] = struct{}{}
	}
	var ans int
	for num := range mp {
		//是第一个
		if _, ok := mp[num-1]; !ok {
			k := num + 1
			for {
				if _, ok := mp[k]; !ok {
					break
				}
				k++
			}
			ans = max(ans, k-num)
		}
	}
	return ans
}

// 283. 移动零
func moveZeroes(nums []int) {
	var index int
	for i := 0; i < len(nums); i++ {
		if nums[i] != 0 {
			nums[i], nums[index] = nums[index], nums[i]
			index++
		}
	}
}

// 11. 盛最多水的容器
func maxArea(height []int) int {
	left, right := 0, len(height)-1
	var ans int
	for left < right {
		ans = max(ans, min(height[left], height[right])*(right-left))
		if height[left] <= height[right] {
			left++
		} else {
			right--
		}
	}
	return ans
}

// 42. 接雨水
func trap(height []int) int {
	left := make([]int, len(height))
	for i := 1; i < len(height); i++ {
		left[i] = max(left[i-1], height[i-1])
	}
	right := make([]int, len(height))
	for i := len(height) - 2; i >= 0; i-- {
		right[i] = max(right[i+1], height[i+1])
	}
	var ans int
	for i := 0; i < len(height); i++ {
		rain := max(0, min(left[i], right[i])-height[i])
		ans += rain
	}
	return ans
}

// 438. 找到字符串中所有字母异位词
func findAnagrams(s string, p string) (ans []int) {
	if len(s) < len(p) {
		return
	}
	target := make(map[byte]int)
	for _, ch := range p {
		target[byte(ch)]++
	}
	//向右一个 收集答案 弹出一个
	for i := 0; i < len(p)-1; i++ {
		target[s[i]]--
		if target[s[i]] == 0 {
			delete(target, s[i])
		}
	}
	for i := 0; i <= len(s)-len(p); i++ {
		target[s[i+len(p)-1]]--
		if target[s[i+len(p)-1]] == 0 {
			delete(target, s[i+len(p)-1])
		}
		if len(target) == 0 {
			ans = append(ans, i)
		}
		target[s[i]]++
		if target[s[i]] == 0 {
			delete(target, s[i])
		}
	}
	return
}

// 239. 滑动窗口最大值
func maxSlidingWindow(nums []int, k int) (ans []int) {
	//维护滑动窗口内的最大值 即维护一个单调减的单调队列
	queue := make([]int, 0, len(nums))
	//扩一个 收集答案 弹出一个
	for i := 0; i < k-1; i++ {
		for len(queue) > 0 && nums[queue[len(queue)-1]] <= nums[i] {
			queue = queue[:len(queue)-1]
		}
		queue = append(queue, i)
	}
	for i := 0; i <= len(nums)-k; i++ {
		for len(queue) > 0 && nums[queue[len(queue)-1]] <= nums[i+k-1] {
			queue = queue[:len(queue)-1]
		}
		queue = append(queue, i+k-1)

		ans = append(ans, nums[queue[0]])
		if queue[0] == i {
			queue = queue[1:]
		}
	}
	return
}

// 76. 最小覆盖子串
func minWindow(s string, t string) string {
	if len(s) < len(t) {
		return ""
	}
	target := make(map[byte]int)
	for _, ch := range t {
		target[byte(ch)]++
	}
	targetSatisfaction := len(target)

	var left, satisfy int
	ans := s + " "

	for right := 0; right < len(s); right++ {
		//本来需求为1 现在变为0 说明删除了一个需要的元素
		target[s[right]]--
		if target[s[right]] == 0 {
			satisfy++
		}
		for ; satisfy == targetSatisfaction; left++ {
			if right-left+1 < len(ans) {
				ans = s[left : right+1]
			}
			//原来不需要这个元素 左窗口滑动之后就需要了 注意先后顺序
			if target[s[left]] == 0 {
				satisfy--
			}
			target[s[left]]++
		}
	}
	if len(ans) > len(s) {
		return ""
	}
	return ans
}

// 41. 缺失的第一个正数
func firstMissingPositive(nums []int) int {
	for i := 0; i < len(nums); i++ {
		for nums[i] > 0 && nums[i] <= len(nums) && nums[i] != nums[nums[i]-1] {
			nums[i], nums[nums[i]-1] = nums[nums[i]-1], nums[i]
		}
	}
	for i := 0; i < len(nums); i++ {
		if nums[i] != i+1 {
			return i + 1
		}
	}
	return len(nums) + 1
}

// 73. 矩阵置零 非常数空间版本
//func setZeroes(matrix [][]int) {
//	zeroRow := make(map[int]struct{})
//	zeroCol := make(map[int]struct{})
//	for i := 0; i < len(matrix); i++ {
//		for j := 0; j < len(matrix[i]); j++ {
//			if matrix[i][j] == 0 {
//				zeroRow[i] = struct{}{}
//				zeroCol[j] = struct{}{}
//			}
//		}
//	}
//	for i := 0; i < len(matrix); i++ {
//		for j := 0; j < len(matrix[i]); j++ {
//			_, okRow := zeroRow[i]
//			_, okCol := zeroCol[j]
//			if okRow || okCol {
//				matrix[i][j] = 0
//			}
//		}
//	}
//}

// 73. 矩阵置零 常数空间版本 使用矩阵的第一行第一列标记是否需要置为0
func setZeroes(matrix [][]int) {
	n, m := len(matrix), len(matrix[0])

	//两个变量标记第一行和第一列是否含有0
	row0, col0 := false, false
	for _, v := range matrix[0] {
		if v == 0 {
			row0 = true
			break
		}
	}
	for _, r := range matrix {
		if r[0] == 0 {
			col0 = true
			break
		}
	}

	//如果一个位置为0 则该行该列都需要置为0
	for i := 1; i < n; i++ {
		for j := 1; j < m; j++ {
			if matrix[i][j] == 0 {
				matrix[i][0] = 0
				matrix[0][j] = 0
			}
		}
	}

	//如果标记位为0 则需要置为0
	for i := 1; i < n; i++ {
		for j := 1; j < m; j++ {
			if matrix[i][0] == 0 || matrix[0][j] == 0 {
				matrix[i][j] = 0
			}
		}
	}

	//最后处理第一行第一列是否需要全部置为0
	if row0 {
		for j := 0; j < m; j++ {
			matrix[0][j] = 0
		}
	}
	if col0 {
		for _, r := range matrix {
			r[0] = 0
		}
	}
}

// 240. 搜索二维矩阵 II
//func searchMatrix(matrix [][]int, target int) bool {
//	for _, row := range matrix {
//		if index, ok := slices.BinarySearch(row, target); ok {
//			return true
//		} else if index == 0 { //小于当前行的首元素 因此肯定小于下面的每一行的每一个元素
//			return false
//		}
//	}
//	return false
//}

// 24. 两两交换链表中的节点 递归
//
//	func swapPairs(head *ListNode) *ListNode {
//		var dfs func(node *ListNode) *ListNode
//		dfs = func(node *ListNode) *ListNode {
//			if node == nil || node.Next == nil {
//				return node
//			}
//			nxt := node.Next
//			node.Next = dfs(nxt.Next)
//			nxt.Next = node
//			return nxt
//		}
//		return dfs(head)
//	}
//
// 24. 两两交换链表中的节点 迭代版本
func swapPairs(head *ListNode) *ListNode {
	dummy := &ListNode{Next: head}
	pre, cur := dummy, dummy.Next
	for cur != nil && cur.Next != nil {
		nxt := cur.Next
		cur.Next = nxt.Next
		nxt.Next = cur
		pre.Next = nxt
		pre = cur
		cur = cur.Next
	}
	return dummy.Next
}

type Node struct {
	Val    int
	Next   *Node
	Random *Node
}

// 138. 随机链表的复制
func copyRandomList(head *Node) *Node {
	mapping := make(map[*Node]*Node)
	for cur := head; cur != nil; cur = cur.Next {
		mapping[cur] = &Node{Val: cur.Val}
	}
	newHead := mapping[head]
	p := newHead
	for cur := head; cur != nil; cur = cur.Next {
		p.Next = mapping[cur.Next]
		p.Random = mapping[cur.Random]
		p = p.Next
	}
	return newHead
}

// 108. 将有序数组转换为二叉搜索树
func sortedArrayToBST(nums []int) *TreeNode {
	if len(nums) <= 0 {
		return nil
	}
	middle := len(nums) / 2
	root := &TreeNode{
		Val:   nums[middle],
		Left:  sortedArrayToBST(nums[:middle]),
		Right: sortedArrayToBST(nums[middle+1:]),
	}
	return root
}

// 230. 二叉搜索树中第 K 小的元素
func kthSmallest(root *TreeNode, k int) int {
	//如果左子树的节点数量==k-1 返回当前节点
	//如果左子树的节点数量<k-1 递归左子树
	//如果左子树的节点数量>k-1 递归右子树
	var nodeCount func(node *TreeNode) int
	nodeCount = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		left, right := nodeCount(node.Left), nodeCount(node.Right)
		return left + right + 1
	}
	var dfs func(node *TreeNode, k int) int
	dfs = func(node *TreeNode, k int) int {
		leftCount := nodeCount(node.Left)
		if leftCount == k-1 {
			return node.Val
		}
		if leftCount < k-1 {
			return dfs(node.Right, k-leftCount-1)
		}
		return dfs(node.Left, k)
	}
	return dfs(root, k)
}

// 199. 二叉树的右视图
func rightSideView(root *TreeNode) (ans []int) {
	curDepth := -1
	var dfs func(node *TreeNode, depth int)
	dfs = func(node *TreeNode, depth int) {
		if node == nil {
			return
		}
		if depth > curDepth {
			curDepth = depth
			ans = append(ans, node.Val)
		}
		dfs(node.Right, depth+1)
		dfs(node.Left, depth+1)
	}
	dfs(root, 0)
	return
}

// 二叉树展开为链表
func flatten(root *TreeNode) {
	var dfs func(node *TreeNode) *TreeNode
	var findMostRight func(node *TreeNode) *TreeNode
	dfs = func(node *TreeNode) *TreeNode {
		if node == nil || (node.Left == nil && node.Right == nil) {
			return node
		}
		left, right := dfs(node.Left), dfs(node.Right)
		node.Right = left
		mostRight := findMostRight(node)
		mostRight.Right = right
		node.Left = nil
		return node
	}
	findMostRight = func(node *TreeNode) *TreeNode {
		for node != nil && node.Right != nil {
			node = node.Right
		}
		return node
	}
	dfs(root)
}

// 105. 从前序与中序遍历序列构造二叉树
func buildTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 {
		return nil
	}
	root := &TreeNode{Val: preorder[0]}
	index := slices.Index(inorder, root.Val)
	left := buildTree(preorder[1:index+1], inorder[:index])
	right := buildTree(preorder[index+1:], inorder[index+1:])
	root.Left = left
	root.Right = right
	return root
}

// 22. 括号生成
func generateParenthesis(n int) (ans []string) {
	//右括号的次数不能大于左括号次数
	var dfs func(left, right int, path []byte)
	dfs = func(left, right int, path []byte) {
		if left == n && right == n {
			ans = append(ans, string(path))
			return
		}
		//左括号
		if left < n {
			path = append(path, '(')
			dfs(left+1, right, path)
			path = path[:len(path)-1]
		}
		//右括号
		if right < left && right < n {
			path = append(path, ')')
			dfs(left, right+1, path)
			path = path[:len(path)-1]
		}
	}
	dfs(0, 0, []byte{})
	return
}

// 79. 单词搜索
func exist(board [][]byte, word string) bool {
	directions := [][]int{
		{0, 1}, {0, -1}, {-1, 0}, {1, 0},
	}
	visited := make([][]bool, len(board))
	for i := 0; i < len(visited); i++ {
		visited[i] = make([]bool, len(board[i]))
	}
	var dfs func(i, j int, index int) bool
	dfs = func(i, j int, index int) bool {
		if index == len(word) {
			return true
		}
		if i < 0 || i >= len(board) || j < 0 || j >= len(board[0]) || visited[i][j] {
			return false
		}
		if board[i][j] == word[index] {
			visited[i][j] = true
			for _, d := range directions {
				if dfs(i+d[0], j+d[1], index+1) {
					return true
				}
			}
			visited[i][j] = false
		}
		return false
	}
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[i]); j++ {
			if board[i][j] == word[0] {
				if dfs(i, j, 0) {
					return true
				}
			}
		}
	}
	return false
}

// 35. 搜索插入位置
func searchInsert(nums []int, target int) int {
	left, right := 0, len(nums)-1
	ans := len(nums)
	for left <= right {
		mid := (right-left)/2 + left
		if nums[mid] == target {
			return mid
		}
		if nums[mid] < target {
			left = mid + 1
		} else {
			ans = mid
			right = mid - 1
		}
	}
	return ans
}

// 74. 搜索二维矩阵
func searchMatrix(matrix [][]int, target int) bool {
	for _, row := range matrix {
		//找到第一个大于等于的
		left, right := 0, len(row)-1
		for left <= right {
			mid := (right-left)/2 + left
			if row[mid] >= target {
				right = mid - 1
			} else {
				left = mid + 1
			}
		}
		if left < len(row) && row[left] == target {
			return true
		}
		if left != len(row) {
			return false
		}
	}
	return false
}

func searchRange(nums []int, target int) []int {
	//找第一个大于等于的：
	left1, right1 := 0, len(nums)-1
	for left1 <= right1 {
		mid := (right1-left1)/2 + left1
		if nums[mid] >= target {
			right1 = mid - 1
		} else {
			left1 = mid + 1
		}
	}
	if left1 == len(nums) || nums[left1] != target {
		return []int{-1, -1}
	}
	left2, right2 := 0, len(nums)-1
	for left2 <= right2 {
		mid := (right2-left2)/2 + left2
		if nums[mid] >= target+1 {
			right2 = mid - 1
		} else {
			left2 = mid + 1
		}
	}
	return []int{left1, left2 - 1}
}

// 171. Excel 表列序号
func titleToNumber(columnTitle string) int {
	var ans int
	base := 1
	for i := len(columnTitle) - 1; i >= 0; i-- {
		ans += base * int(columnTitle[i]-'A'+1)
		base *= 26
	}
	return ans
}

// 168. Excel表列名称
func convertToTitle(columnNumber int) string {
	bytes := make([]byte, 0)
	for columnNumber > 0 {
		columnNumber -= 1
		bytes = append(bytes, 'A'+byte(columnNumber%26))
		columnNumber /= 26
	}
	slices.Reverse(bytes)
	return string(bytes)
}

// 三数之和
func threeSum(nums []int) (ans [][]int) {
	slices.Sort(nums)
	for i := 0; i < len(nums); i++ {
		n1 := nums[i]
		if i == 0 || n1 != nums[i-1] {
			left, right := i+1, len(nums)-1
			for left < right {
				n2, n3 := nums[left], nums[right]
				if n1+n2+n3 < 0 {
					left++
				} else if n1+n2+n3 > 0 {
					right--
				} else {
					ans = append(ans, []int{n1, n2, n3})
					for left < right && nums[left] == n2 {
						left++
					}
					for left < right && nums[right] == n3 {
						right--
					}
				}
			}
		}
	}
	return
}

// 143. 重排链表
// preSlow只有在归并排序中用到了
func reorderList(head *ListNode) {
	//从中间断开 翻转 然后merge
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}

	var reverse func(node *ListNode) *ListNode
	reverse = func(node *ListNode) *ListNode {
		var pre *ListNode
		for cur := node; cur != nil; {
			nxt := cur.Next
			cur.Next = pre
			cur, pre = nxt, cur
		}
		return pre
	}
	reversed := reverse(slow.Next)
	slow.Next = nil
	var merge func(l1, l2 *ListNode)
	merge = func(l1, l2 *ListNode) {
		for l1 != nil && l2 != nil {
			nxt := l2.Next
			l2.Next = l1.Next
			l1.Next = l2
			l1 = l2.Next
			l2 = nxt
		}
	}
	merge(head, reversed)
}

// 62. 不同路径
func uniquePaths(m int, n int) int {
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
	}
	for i := 0; i < m; i++ {
		dp[i][0] = 1
	}
	for i := 0; i < n; i++ {
		dp[0][i] = 1
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[i][j] = dp[i-1][j] + dp[i][j-1]
		}
	}
	return dp[m-1][n-1]
}

// 64. 最小路径和
func minPathSum(grid [][]int) int {
	for i := 1; i < len(grid); i++ {
		grid[i][0] += grid[i-1][0]
	}
	for i := 1; i < len(grid[0]); i++ {
		grid[0][i] += grid[0][i-1]
	}
	for i := 1; i < len(grid); i++ {
		for j := 1; j < len(grid[i]); j++ {
			grid[i][j] += min(grid[i-1][j], grid[i][j-1])
		}
	}
	return grid[len(grid)-1][len(grid[0])-1]
}

// 1143. 最长公共子序列
func longestCommonSubsequence(text1 string, text2 string) int {
	dp := make([][]int, len(text1))
	for i := 0; i < len(text1); i++ {
		dp[i] = make([]int, len(text2))
	}
	for i := 0; i < len(text1); i++ {
		if text1[i] == text2[0] {
			for j := i; j < len(text1); j++ {
				dp[j][0] = 1
			}
		}
	}
	for i := 0; i < len(text2); i++ {
		if text2[i] == text1[0] {
			for j := i; j < len(text2); j++ {
				dp[0][j] = 1
			}
		}
	}
	for i := 1; i < len(text1); i++ {
		for j := 1; j < len(text2); j++ {
			if text1[i] == text2[j] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
			}
		}
	}
	return dp[len(text1)-1][len(text2)-1]
}

// 72. 编辑距离
func minDistance(word1 string, word2 string) int {
	m, n := len(word1), len(word2)
	dp := make([][]int, m+1)
	for i := 0; i < m+1; i++ {
		dp[i] = make([]int, n+1)
	}
	for i := 1; i < m+1; i++ {
		dp[i][0] = i
	}
	for i := 1; i < n+1; i++ {
		dp[0][i] = i
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if word1[i] == word2[j] {
				dp[i+1][j+1] = dp[i][j]
			} else {
				dp[i+1][j+1] = min(dp[i][j+1]+1, dp[i+1][j]+1, dp[i][j]+1)
			}
		}
	}
	return dp[m][n]
}

// 124. 二叉树中的最大路径和
func maxPathSum(root *TreeNode) int {
	var ans = math.MinInt
	var dfs func(node *TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		left, right := dfs(node.Left), dfs(node.Right)
		ans = max(ans, left+right+node.Val, left+node.Val, right+node.Val, node.Val)
		return max(left+node.Val, right+node.Val, node.Val)
	}
	dfs(root)
	return ans
}

// 739. 每日温度 单调栈 寻找下一个更高的 维护单调减的单调栈
func dailyTemperatures(temperatures []int) []int {
	stack := make([]int, 0, len(temperatures))
	ans := make([]int, len(temperatures))
	for i := 0; i < len(temperatures); i++ {
		for len(stack) > 0 && temperatures[stack[len(stack)-1]] < temperatures[i] {
			ans[stack[len(stack)-1]] = i - stack[len(stack)-1]
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, i)
	}
	return ans
}

type FrequencyHeap [][2]int

func (f *FrequencyHeap) Len() int {
	return len(*f)
}

func (f *FrequencyHeap) Less(i, j int) bool {
	return (*f)[i][1] < (*f)[j][1]
}

func (f *FrequencyHeap) Swap(i, j int) {
	(*f)[i], (*f)[j] = (*f)[j], (*f)[i]
}

func (f *FrequencyHeap) Push(x any) {
	*f = append(*f, x.([2]int))
}

func (f *FrequencyHeap) Pop() any {
	x := (*f)[f.Len()-1]
	*f = (*f)[:f.Len()-1]
	return x
}

// 347. 前 K 个高频元素 出现频率前k高
func topKFrequent(nums []int, k int) (ans []int) {
	occur := make(map[int]int)
	//先统计一遍全部元素的出现频率
	for _, num := range nums {
		occur[num]++
	}
	hp := &FrequencyHeap{}
	for num, times := range occur {
		heap.Push(hp, [2]int{num, times})
		if hp.Len() > k {
			heap.Pop(hp)
		}
	}
	for _, p := range *hp {
		ans = append(ans, p[0])
	}
	return
}

func add(num1 string, num2 string) string {
	ans := make([]byte, 0, len(num1)+len(num2))
	bytes1 := []byte(num1)
	bytes2 := []byte(num2)
	slices.Reverse(bytes1)
	slices.Reverse(bytes2)
	carry := 0
	var i, j int
	for i < len(bytes1) || j < len(bytes2) || carry != 0 {
		var val int
		if i < len(bytes1) {
			val += int(bytes1[i] - '0')
			i++
		}
		if j < len(bytes2) {
			val += int(bytes2[j] - '0')
			j++
		}
		if carry != 0 {
			val += carry
		}
		carry = val / 10
		val %= 10
		ans = append(ans, byte(val+'0'))
	}
	slices.Reverse(ans)
	return string(ans)
}

func multiplyByBit(num1 string, digit byte, zeroCount int) string {
	//zeroCount表示在结果后面加几位0
	bytes := []byte(num1)
	ans := make([]byte, 0, len(num1))
	slices.Reverse(bytes)
	carry := 0
	digitInt := int(digit - '0')
	for i := 0; i < len(bytes) || carry != 0; i++ {
		var val int
		if i < len(bytes) {
			val += digitInt * int(bytes[i]-'0')
		}
		if carry != 0 {
			val += carry
		}
		carry = val / 10
		val %= 10
		ans = append(ans, byte(val+'0'))
	}
	slices.Reverse(ans)
	for i := 0; i < zeroCount; i++ {
		ans = append(ans, '0')
	}
	return string(ans)
}

// 43. 字符串相乘
func multiply(num1 string, num2 string) string {
	if num1 == "0" || num2 == "0" {
		return "0"
	}
	temp := make([]string, 0, len(num2))
	for i := len(num2) - 1; i >= 0; i-- {
		temp = append(temp, multiplyByBit(num1, num2[i], len(num2)-i-1))
	}
	ans := temp[0]
	for i := 1; i < len(temp); i++ {
		ans = add(ans, temp[i])
	}
	return ans
}

// 70. 爬楼梯
//
//	func climbStairs(n int) int {
//		if n <= 2 {
//			return n
//		}
//		dp := make([]int, n+1)
//		dp[1] = 1
//		dp[2] = 2
//		for i := 3; i <= n; i++ {
//			dp[i] = dp[i-1] + dp[i-2]
//		}
//		return dp[n]
//	}
//
// 70. 爬楼梯
func climbStairs(n int) int {
	if n <= 2 {
		return n
	}
	dp1 := 1
	dp2 := 2
	for i := 3; i <= n; i++ {
		dp1, dp2 = dp2, dp1+dp2
	}
	return dp2
}

// 118. 杨辉三角
func generate(numRows int) (ans [][]int) {
	if numRows == 1 {
		return [][]int{{1}}
	}
	if numRows == 2 {
		return [][]int{{1}, {1, 1}}
	}
	ans = append(ans, []int{1}, []int{1, 1})
	for i := 3; i <= numRows; i++ {
		row := make([]int, i)
		row[0] = 1
		row[len(row)-1] = 1
		for j := 1; j < len(row)-1; j++ {
			row[j] = ans[i-2][j-1] + ans[i-2][j]
		}
		ans = append(ans, row)
	}
	return
}

func rob(nums []int) int {
	if len(nums) == 1 {
		return nums[0]
	}
	dp := make([]int, len(nums))
	dp[0] = nums[0]
	dp[1] = max(nums[0], nums[1])
	for i := 2; i < len(nums); i++ {
		dp[i] = max(dp[i-1], dp[i-2]+nums[i])
	}
	return dp[len(nums)-1]
}

// 279. 完全平方数
func numSquares(n int) int {
	if n == 1 {
		return 1
	}
	dp := make([]int, n+1)
	dp[0] = 0
	dp[1] = 1
	for i := 2; i <= n; i++ {
		dp[i] = math.MaxInt
		for j := 1; j*j <= i; j++ {
			if dp[i-j*j] != math.MaxInt {
				dp[i] = min(dp[i], dp[i-j*j]+1)
			}
		}
	}
	return dp[n]
}

// 322. 零钱兑换 完全背包问题
func coinChange(coins []int, amount int) int {
	dp := make([]int, amount+1)
	for i := 1; i < len(dp); i++ {
		dp[i] = amount + 1
	}
	for i := 0; i < len(coins); i++ {
		for j := coins[i]; j <= amount; j++ {
			dp[j] = min(dp[j], dp[j-coins[i]]+1)
		}
	}
	if dp[amount] == amount+1 {
		return -1
	}
	return dp[amount]
}
