package main

import (
	"fmt"
	"math"
	"math/rand/v2"
	"net"
	"slices"
	"strconv"
	"strings"
)

// 5. 最长回文子串
func longestPalindrome(s string) string {
	n := len(s)
	dp := make([][]bool, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]bool, n)
	}
	ans := ""
	for i := n - 1; i >= 0; i-- {
		for j := i; j < n; j++ {
			if s[i] == s[j] {
				if i == j || j-i == 1 {
					dp[i][j] = true
				} else {
					dp[i][j] = dp[i+1][j-1]
				}
				if dp[i][j] && j-i+1 > len(ans) {
					ans = s[i : j+1]
				}
			}
		}
	}
	return ans
}

// 1143. 最长公共子序列
//func longestCommonSubsequence(text1 string, text2 string) int {
//	//dpij 以i-1 j-1为结尾的最长公共子序列 这样创建dp数组不需要额外的初始化
//	dp := make([][]int, len(text1)+1)
//	for i := 0; i < len(dp); i++ {
//		dp[i] = make([]int, len(text2)+1)
//	}
//	for i := 0; i < len(text1); i++ {
//		for j := 0; j < len(text2); j++ {
//			if text1[i] == text2[j] {
//				dp[i+1][j+1] = dp[i][j] + 1
//			} else {
//				dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
//			}
//		}
//	}
//	return dp[len(text1)][len(text2)]
//}

// 763. 划分字母区间
func partitionLabels(s string) (ans []int) {
	//统计每一个字符出现的最晚位置 遇到一个字符就向右扩充到最晚位置
	lastAppear := make(map[byte]int)
	for loc, ch := range s {
		lastAppear[byte(ch)] = loc
	}
	var maxRight, j int
	for i := 0; i < len(s); i++ {
		maxRight = max(maxRight, lastAppear[s[i]])
		j++
		if i == maxRight { //完成一个片段的收集
			ans = append(ans, j)
			j = 0
		}
	}
	return
}

// 39. 组合总和
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

func merge2(nums1 []int, m int, nums2 []int, n int) {
	i, j := m-1, n-1
	for index := m + n - 1; index >= 0; index-- {
		if i >= 0 && j >= 0 {
			if nums1[i] >= nums2[j] {
				nums1[index] = nums1[i]
				i--
			} else {
				nums1[index] = nums2[j]
				j--
			}
		} else if i >= 0 {
			nums1[index] = nums1[i]
			i--
		} else {
			nums1[index] = nums2[j]
			j--
		}
	}
}

//type Solution struct {
//	nodes []*ListNode
//}

//func Constructor(head *ListNode) Solution {
//	nodes := make([]*ListNode, 0)
//	for cur := head; cur != nil; cur = cur.Next {
//		nodes = append(nodes, cur)
//	}
//	return Solution{nodes: nodes}
//}

//func (this *Solution) GetRandom() int {
//	return this.nodes[rand.IntN(len(this.nodes))].Val
//}

// 547. 省份数量
func findCircleNum(isConnected [][]int) int {
	n := len(isConnected)
	father := make([]int, n)
	for i := 0; i < len(father); i++ {
		father[i] = i
	}
	var merge func(i, j int) bool
	var find func(i int) int
	find = func(i int) int {
		if father[i] != i {
			father[i] = find(father[i])
		}
		return father[i]
	}
	merge = func(x, y int) bool {
		fx, fy := find(x), find(y)
		if fx != fy {
			father[fx] = fy
			return true
		}
		return false
	}
	var ans = n
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if isConnected[i][j] == 1 {
				if merge(i, j) {
					ans--
				}
			}
		}
	}
	return ans
}

// 234. 回文链表
func isPalindrome(head *ListNode) bool {
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
			pre, cur = cur, nxt
		}
		return pre
	}
	for p, q := head, reverse(slow); p != nil && q != nil; p, q = p.Next, q.Next {
		if p.Val != q.Val {
			return false
		}
	}
	return true
}

func findTargetSumWays(nums []int, target int) int {
	var sum int
	for _, num := range nums {
		sum += num
	}
	diff := sum - target
	if diff%2 != 0 {
		return 0
	}
	bag := diff / 2
	dp := make([][]int, len(nums))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, bag+1)
	}
	dp[0][0] += 1
	if nums[0] <= bag {
		dp[0][nums[0]] += 1
	}
	for i := 1; i < len(nums); i++ {
		for j := 0; j <= bag; j++ {
			dp[i][j] = dp[i-1][j]
			if j >= nums[i] {
				dp[i][j] += dp[i-1][j-nums[i]]
			}
		}
	}
	return dp[len(nums)-1][bag]
}

//	type Entry struct {
//		Key, Value int
//	}
//
//	type DoubleNode struct {
//		Entry *Entry
//		Next  *DoubleNode
//		Prev  *DoubleNode
//	}
//
//	type DoubleList struct {
//		DummyHead, DummyTail *DoubleNode
//	}
//
//	func NewDoubleList() *DoubleList {
//		dummyHead := &DoubleNode{}
//		dummyTail := &DoubleNode{}
//		dummyHead.Next = dummyTail
//		dummyTail.Prev = dummyHead
//		return &DoubleList{
//			DummyHead: dummyHead,
//			DummyTail: dummyTail,
//		}
//	}
//
//	func (l *DoubleList) MoveToFront(node *DoubleNode) {
//		//先把他删除 再移动到头部
//		l.RemoveNode(node)
//		l.PushFront(node)
//	}
//
//	func (l *DoubleList) RemoveNode(node *DoubleNode) {
//		node.Next.Prev = node.Prev
//		node.Prev.Next = node.Next
//		node.Prev = nil
//		node.Next = nil
//	}
//
//	func (l *DoubleList) PushFront(node *DoubleNode) {
//		node.Next = l.DummyHead.Next
//		l.DummyHead.Next.Prev = node
//		l.DummyHead.Next = node
//		node.Prev = l.DummyHead
//	}
//
//	func (l *DoubleList) Last() *DoubleNode {
//		return l.DummyTail.Prev
//	}
//
//	type LRUCache struct {
//		List      *DoubleList
//		KeyToNode map[int]*DoubleNode
//		Capacity  int
//	}
//
//	func Constructor(capacity int) LRUCache {
//		list := NewDoubleList()
//		keyToNode := make(map[int]*DoubleNode)
//		return LRUCache{
//			List:      list,
//			KeyToNode: keyToNode,
//			Capacity:  capacity,
//		}
//	}
//
//	func (this *LRUCache) Get(key int) int {
//		if node, ok := this.KeyToNode[key]; ok {
//			//移动到最前面
//			this.List.MoveToFront(node)
//			return node.Entry.Value
//		} else {
//			return -1
//		}
//	}
//
//	func (this *LRUCache) Put(key int, value int) {
//		if node, ok := this.KeyToNode[key]; ok {
//			node.Entry.Value = value
//			this.List.MoveToFront(node)
//		} else {
//			newNode := &DoubleNode{
//				Entry: &Entry{
//					Key:   key,
//					Value: value,
//				},
//			}
//			this.List.PushFront(newNode)
//			this.KeyToNode[key] = newNode
//			if len(this.KeyToNode) > this.Capacity {
//				//从最后删除一个
//				lastNode := this.List.Last()
//				this.List.RemoveNode(lastNode)
//				delete(this.KeyToNode, lastNode.Entry.Key)
//			}
//		}
//	}
//

func shuffle(nums []int) []int {
	for i := 0; i < len(nums); i++ {
		index := rand.IntN(len(nums)-i) + i
		nums[i], nums[index] = nums[index], nums[i]
	}
	return nums
}

//	type Solution struct {
//		original []int
//	}
//
//	func Constructor(nums []int) Solution {
//		return Solution{
//			original: nums,
//		}
//	}
//
//	func (this *Solution) Reset() []int {
//		return this.original
//	}
//
//	func (this *Solution) Shuffle() []int {
//		shuffled := make([]int, len(this.original))
//		copy(shuffled, this.original)
//		n := len(shuffled)
//		for i := 0; i < n; i++ {
//			index := rand.IntN(n-i) + i
//			shuffled[index], shuffled[i] = shuffled[i], shuffled[index]
//		}
//		return shuffled
//	}
//
// 958. 二叉树的完全性检验
// func isCompleteTree(root *TreeNode) bool {
//
// }
// 面试题 17.14. 最小K个数
func smallestK(arr []int, k int) []int {
	var helper func(left, right, k int) int
	helper = func(left, right, k int) int {
		if left >= right {
			return arr[k]
		}
		pivot := arr[rand.IntN(right-left+1)+left]
		i, j := left, right
		for i <= j {
			for arr[i] < pivot {
				i++
			}
			for arr[j] > pivot {
				j--
			}
			if i <= j {
				arr[i], arr[j] = arr[j], arr[i]
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
	if k == 0 {
		return []int{}
	}
	helper(0, len(arr)-1, k)
	return arr[:k]
}

// 58. 最后一个单词的长度
func lengthOfLastWord(s string) int {
	//从后向前 从第一个非空格遍历到第一个空格
	i := len(s) - 1
	for ; i >= 0; i-- {
		if s[i] != ' ' {
			for j := i - 1; j >= 0; j-- {
				if s[j] == ' ' {
					return i - j
				}
			}
			return i + 1
		}
	}
	return i + 1
}

// 958. 二叉树的完全性检验 在层序遍历一个二叉树的时候，一个非空节点之前不能有空节点
//
//	func isCompleteTree(root *TreeNode) bool {
//		queue := make([]*TreeNode, 0)
//		queue = append(queue, root)
//		empty := false
//		for len(queue) > 0 {
//			x := queue[0]
//			queue = queue[1:]
//			if x == nil {
//				empty = true
//			} else {
//				if empty {
//					return false
//				}
//				queue = append(queue, x.Left)
//				queue = append(queue, x.Right)
//			}
//		}
//		return true
//	}
//
// 958. 二叉树的完全性检验 编号不能超过节点总数
func isCompleteTree(root *TreeNode) bool {
	var nodeCount func(node *TreeNode) int
	nodeCount = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		return 1 + nodeCount(node.Left) + nodeCount(node.Right)
	}
	count := nodeCount(root)
	var dfs func(node *TreeNode, index int) bool
	dfs = func(node *TreeNode, index int) bool {
		if node == nil {
			return true
		}
		if index > count {
			return false
		}
		return dfs(node.Left, index*2) && dfs(node.Right, index*2+1)
	}
	return dfs(root, 1)
}

// 328. 奇偶链表
func oddEvenList(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	odd := head
	even, evenHead := head.Next, head.Next
	for even != nil && even.Next != nil {
		odd.Next = even.Next
		odd = odd.Next
		even.Next = odd.Next
		even = even.Next
	}
	odd.Next = evenHead
	return head
}

// 523. 连续的子数组和
// func checkSubarraySum(nums []int, k int) bool {
//
// }
// 128. 最长连续序列
func longestConsecutive2(nums []int) int {
	mp := make(map[int]bool)
	for _, num := range nums {
		mp[num] = true
	}
	var ans int
	for k, _ := range mp {
		if !mp[k-1] { //是序列中的第一个数
			j := k + 1
			for mp[j] {
				j++
			}
			ans = max(ans, j-k)
		}
	}
	return ans
}

// 最多可以完成两笔交易 且不能同时参与多笔交易
// 状态机：初始状态0 第一次持有1 第一次卖出2 第二次持有3 第二次卖出4
func maxProfit3(prices []int) int {
	dp := make([][5]int, len(prices))
	dp[0][1] = -prices[0]
	dp[0][3] = -prices[0]
	for i := 1; i < len(prices); i++ {
		//第一次持有：之前就第一次持有 未持有状态下买入
		dp[i][1] = max(dp[i-1][1], dp[i-1][0]-prices[i])
		//第一次卖出：之间就已经卖出 之前第一次持有状态下卖出
		dp[i][2] = max(dp[i-1][2], dp[i-1][1]+prices[i])
		//第二次持有：
		dp[i][3] = max(dp[i-1][3], dp[i-1][2]-prices[i])
		dp[i][4] = max(dp[i-1][4], dp[i-1][3]+prices[i])
	}
	return dp[len(dp)-1][4]
}

// 415. 字符串相加
func addStrings(num1 string, num2 string) string {
	ans := make([]byte, 0, len(num1))
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

// 82. 删除排序链表中的重复元素 II 重复元素不保留
//
//	func deleteDuplicates(head *ListNode) *ListNode {
//		if head == nil || head.Next == nil {
//			return head
//		}
//		if head.Val == head.Next.Val {
//			p := head.Next.Next
//			for p != nil && p.Val == head.Val {
//				p = p.Next
//			}
//			return deleteDuplicates(p)
//		} else {
//			head.Next = deleteDuplicates(head.Next)
//			return head
//		}
//	}
//
// 82. 删除排序链表中的重复元素 II
func deleteDuplicates(head *ListNode) *ListNode {
	dummy := &ListNode{Next: head}
	cur := dummy
	for cur.Next != nil && cur.Next.Next != nil {
		if cur.Next.Val == cur.Next.Next.Val {
			p := cur.Next.Next.Next
			for p != nil && p.Val == cur.Next.Val {
				p = p.Next
			}
			cur.Next = p
		} else {
			cur = cur.Next
		}
	}
	return dummy.Next
}

// 92. 反转链表 II
func reverseBetween(head *ListNode, left int, right int) *ListNode {
	dummy := &ListNode{Next: head}
	cur := dummy
	var temp *ListNode
	for i := 0; i < left; i++ {
		temp = cur
		cur = cur.Next
	}
	var pre *ListNode
	for i := 0; i <= right-left; i++ {
		nxt := cur.Next
		cur.Next = pre
		pre, cur = cur, nxt
	}
	temp.Next.Next = cur
	temp.Next = pre
	return dummy.Next
}

// 628. 三个数的最大乘积
func maximumProduct(nums []int) int {
	//最大的三个和最小的两个即可
	max1, max2, max3 := math.MinInt, math.MinInt, math.MinInt
	min1, min2 := math.MaxInt, math.MaxInt
	for _, num := range nums {
		if num > max1 {
			max3 = max2
			max2 = max1
			max1 = num
		} else if num > max2 {
			max3 = max2
			max2 = num
		} else if num > max3 {
			max3 = num
		}

		if num < min1 {
			min2 = min1
			min1 = num
		} else if num < min2 {
			min2 = num
		}
	}
	return max(max1*max2*max3, min1*min2*max1)
}

func decodeString(s string) string {
	numTemp := make([]byte, 0, len(s))
	numStack := make([]int, 0, len(s))
	temp := make([]byte, 0, len(s))
	//遇到左括号 记录数字
	//遇到右括号 弹出一个数字 重复
	for _, ch := range s {
		if ch >= '0' && ch <= '9' {
			numTemp = append(numTemp, byte(ch))
		} else if ch >= 'a' && ch <= 'z' {
			temp = append(temp, byte(ch))
		} else if ch == '[' { //收集数字
			num, _ := strconv.Atoi(string(numTemp))
			temp = append(temp, byte(ch))
			numStack = append(numStack, num)
			numTemp = []byte{}
		} else { //遇到右括号
			times := numStack[len(numStack)-1]
			numStack = numStack[:len(numStack)-1]
			i := len(temp) - 1
			for temp[i] != '[' {
				i--
			}
			toAppend := string(temp[i+1:])
			temp = temp[:i]
			for i := 0; i < times; i++ {
				temp = append(temp, toAppend...)
			}
		}
	}
	return string(temp)
}

type Codec struct {
}

//func Constructor() Codec {
//
//}

// Serializes a tree to a single string.
func (this *Codec) serialize(root *TreeNode) string {
	if root == nil {
		return "X"
	}
	return strconv.Itoa(root.Val) + this.serialize(root.Left) + "," + this.serialize(root.Right)
}

// Deserializes your encoded data to tree.
func (this *Codec) deserialize(data string) *TreeNode {
	list := strings.Split(data, ",")
	return BuildTree(&list)
}

func BuildTree(list *[]string) *TreeNode {
	rootVal := (*list)[0]
	if rootVal == "X" {
		return nil
	}
	*list = (*list)[1:]
	val, _ := strconv.Atoi(rootVal)
	root := &TreeNode{Val: val}
	root.Left = BuildTree(list)
	root.Right = BuildTree(list)
	return root
}
func minDiffInBST(root *TreeNode) int {
	var ans = math.MaxInt
	//一个节点和其左子树的最大值 一个节点和其右子树的最小值之间的差异
	//向上返回当前子树的最大值和最小值
	var dfs func(node *TreeNode) (int, int)
	dfs = func(node *TreeNode) (int, int) {
		if node == nil {
			return math.MinInt, math.MaxInt
		}
		leftMin, leftMax := dfs(node.Left)
		rightMin, rightMax := dfs(node.Right)
		ans = min(ans, abs(node.Val-leftMax), abs(node.Val-rightMin))
		return max(leftMax, rightMax, node.Val), min(leftMin, rightMin, node.Val)
	}
	dfs(root)
	return ans
}

func abs(num int) int {
	if num < 0 {
		return -num
	}
	return num
}

func findShortestSubArray(nums []int) int {
	cnt := make(map[int][]int)
	var degree int
	for i, num := range nums {
		if v, ok := cnt[num]; !ok { //第一次出现
			cnt[num] = []int{1, i, i}
		} else {
			v[0] += 1
			v[2] = i
		}
		degree = max(degree, cnt[num][0])
	}
	fmt.Println(degree)
	var ans = len(nums)
	for _, v := range cnt {
		if v[0] == degree {
			ans = min(ans, v[2]-v[1]+1)
		}
	}
	return ans
}
func lengthOfLIS(nums []int) int {
	sequence := make([]int, 0, len(nums))
	for i := 0; i < len(nums); i++ {
		index := SearchInts(sequence, nums[i])
		if index == len(sequence) {
			sequence = append(sequence, nums[i])
		} else {
			sequence[index] = nums[i]
		}
	}
	return len(sequence)
}

func SearchInts(nums []int, target int) int {
	left, right := 0, len(nums)-1
	ans := len(nums)
	for left <= right {
		mid := (right-left)/2 + left
		if nums[mid] >= target {
			right = mid - 1
			ans = mid
		} else {
			left = mid + 1
		}
	}
	return ans
}

func mergeSort(nums []int) []int {
	var helper func(left, right int)
	var merge func(left, mid, right int)
	helper = func(left, right int) {
		if left >= right {
			return
		}
		mid := (right-left)/2 + left
		helper(left, mid)
		helper(mid+1, right)
		merge(left, mid, right)
	}
	merge = func(left, mid, right int) {
		temp := make([]int, 0, right-left+1)
		i, j := left, mid+1
		for i <= mid && j <= right {
			if nums[i] <= nums[j] {
				temp = append(temp, nums[i])
				i++
			} else {
				temp = append(temp, nums[j])
				j++
			}
		}
		for ; i <= mid; i++ {
			temp = append(temp, nums[i])
		}
		for ; j <= right; j++ {
			temp = append(temp, nums[j])
		}
		for i := 0; i < len(temp); i++ {
			nums[i+left] = temp[i]
		}
	}
	helper(0, len(nums)-1)
	return nums
}
func validIPAddress(queryIP string) string {
	ip := net.ParseIP(queryIP)
	if ip == nil {
		return "Neither"
	}
	if ip.To4() != nil {
		for _, s := range strings.Split(queryIP, ".") {
			if len(s) > 1 && s[0] == '0' {
				return "Neither"
			}
		}
		return "IPv4"
	}
	for _, s := range strings.Split(queryIP, ":") {
		if len(s) > 4 || s == "" {
			return "Neither"
		}
	}
	return "IPv6"
}

func mySqrt(x int) float64 {
	left, right := float64(0), float64(x)
	var ans float64 = -1
	for right-left >= 1e-10 {
		mid := (right-left)/2 + left
		if mid*mid <= float64(x) {
			ans = mid
			left = mid
		} else {
			right = mid
		}
	}
	return ans
}
