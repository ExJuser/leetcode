package main

import (
	"bufio"
	"container/heap"
	"math"
	"os"
	"slices"
	"sort"
	"strconv"
	"strings"
)

var scanner = bufio.NewScanner(os.Stdin)

func GetInt() int {
	scanner.Scan()
	out, _ := strconv.Atoi(scanner.Text())
	return out
}

func GetString() string {
	scanner.Scan()
	return scanner.Text()
}

func GetIntSlice() []int {
	out := make([]int, 0, 1000)
	scanner.Scan()
	splits := strings.Split(scanner.Text(), " ")
	for _, numStr := range splits {
		num, _ := strconv.Atoi(numStr)
		out = append(out, num)
	}
	return out
}

func GetStringSlice() []string {
	out := make([]string, 0, 1000)
	scanner.Scan()
	splits := strings.Split(scanner.Text(), " ")
	for _, str := range splits {
		out = append(out, str)
	}
	return out
}
func twoSum(nums []int, target int) []int {
	mp := make(map[int]int)
	for i, num := range nums {
		if j, ok := mp[target-num]; ok {
			return []int{j, i}
		} else {
			mp[num] = i
		}
	}
	return nil
}

func groupAnagrams(strs []string) (ans [][]string) {
	mp := make(map[string][]string)
	for _, str := range strs {
		bytes := []byte(str)
		slices.Sort(bytes)
		mp[string(bytes)] = append(mp[string(bytes)], str)
	}
	for _, v := range mp {
		ans = append(ans, v)
	}
	return
}

func longestConsecutive(nums []int) int {
	mp := make(map[int]bool)
	for _, num := range nums {
		mp[num] = true
	}
	ans := 0
	for k, _ := range mp {
		if !mp[k-1] {
			key := k + 1
			for mp[key] {
				key++
			}
			ans = max(ans, key-k)
		}
	}
	return ans
}
func moveZeroes(nums []int) {
	index := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] != 0 {
			nums[index] = nums[i]
			index++
		}
	}
	for i := index; i < len(nums); i++ {
		nums[i] = 0
	}
}

func maxArea(height []int) int {
	left, right := 0, len(height)-1
	area := 0
	for left < right {
		area = max(area, (min(height[left], height[right]))*(right-left))
		if height[left] <= height[right] {
			left++
		} else {
			right--
		}
	}
	return area
}

func threeSum(nums []int) (ans [][]int) {
	slices.Sort(nums)
	for i, n1 := range nums {
		if (i == 0 || n1 != nums[i-1]) && n1 <= 0 {
			left, right := i+1, len(nums)-1
			for left < right {
				n2, n3 := nums[left], nums[right]
				sum := n1 + n2 + n3
				if sum > 0 {
					right--
				} else if sum < 0 {
					left++
				} else {
					ans = append(ans, []int{n1, nums[left], nums[right]})
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

func subarraySum(nums []int, k int) int {
	hash := map[int]int{0: 1}
	prefixSum := 0
	count := 0
	for _, num := range nums {
		prefixSum += num
		count += hash[prefixSum-k]
		hash[prefixSum]++
	}
	return count
}

/*
*
峰值元素是指其值严格大于左右相邻值的元素。
给你一个整数数组 nums，找到峰值元素并返回其索引。
数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 所在位置即可。
你可以假设 nums[-1] = nums[n] = -∞ 。
你必须实现时间复杂度为 O(log n) 的算法来解决此问题。
对于所有有效的 i 都有 nums[i] != nums[i + 1]
*/
func findPeakElement(nums []int) int {
	if len(nums) == 1 || nums[0] > nums[1] {
		return 0
	}
	if nums[len(nums)-1] > nums[len(nums)-2] {
		return len(nums) - 1
	}
	left, right := 1, len(nums)-2
	for left <= right {
		mid := (right-left)/2 + left
		if nums[mid] > nums[mid-1] && nums[mid] > nums[mid+1] {
			return mid
		} else if nums[mid-1] > nums[mid] {
			right = mid - 1
		} else if nums[mid+1] > nums[mid] {
			left = mid + 1
		}
	}
	return -1
}

type ListNode struct {
	Next *ListNode
	Val  int
}

/*
*
删除链表的倒数第 n 个结点，并且返回链表的头结点。
有可能删除头结点 因此需要添加dummy head
快慢指针
*/
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummy := &ListNode{Next: head}
	slow := dummy
	fast := dummy
	for i := 0; i < n; i++ {
		fast = fast.Next
	}
	for fast.Next != nil {
		slow = slow.Next
		fast = fast.Next
	}
	slow.Next = slow.Next.Next
	return dummy.Next
}

/*
*
将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
*/
func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	dummy := &ListNode{}
	p := dummy
	for list1 != nil && list2 != nil {
		if list1.Val <= list2.Val {
			p.Next = &ListNode{Val: list1.Val}
			list1 = list1.Next
		} else {
			p.Next = &ListNode{Val: list2.Val}
			list2 = list2.Next
		}
		p = p.Next
	}
	if list1 == nil {
		p.Next = list2
	} else {
		p.Next = list1
	}
	return dummy.Next
}

/*
*
给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

另开一个链表的做法
*/
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	dummy := &ListNode{}
	p := dummy
	carry := 0
	for l1 != nil && l2 != nil {
		val := l1.Val + l2.Val + carry
		p.Next = &ListNode{Val: val % 10}
		p = p.Next
		carry = val / 10
		l1, l2 = l1.Next, l2.Next
	}
	var next *ListNode
	if l1 == nil {
		next = l2
	} else {
		next = l1
	}
	for next != nil {
		val := next.Val + carry
		p.Next = &ListNode{Val: val % 10}
		p = p.Next
		carry = val / 10
		next = next.Next
	}
	if carry != 0 {
		p.Next = &ListNode{Val: 1}
	}
	return dummy.Next
}

/*
*
给你一个链表的头节点 head 和一个特定值 x ，请你对链表进行分隔，使得所有 小于 x 的节点都出现在 大于或等于 x 的节点之前。

你应当 保留 两个分区中每个节点的初始相对位置。
*/
//func partition(head *ListNode, x int) *ListNode {
//	leftDummy := &ListNode{}
//	rightDummy := &ListNode{}
//	p, q := leftDummy, rightDummy
//	for ; head != nil; head = head.Next {
//		if head.Val < x {
//			p.Next = &ListNode{Val: head.Val}
//			p = p.Next
//		} else {
//			q.Next = &ListNode{Val: head.Val}
//			q = q.Next
//		}
//	}
//	p.Next = rightDummy.Next
//	return leftDummy.Next
//}

// MinStack
/**
设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。

实现 MinStack 类:

MinStack() 初始化堆栈对象。
void push(int val) 将元素val推入堆栈。
void pop() 删除堆栈顶部的元素。
int top() 获取堆栈顶部的元素。
int getMin() 获取堆栈中的最小元素。
*/
type MinStack struct {
	stack []int
	//额外用一个栈时刻维护当前时刻的最小值
	//即使伴随着主栈元素出栈
	minStack []int
}

func Constructor() MinStack {
	return MinStack{
		stack:    make([]int, 0),
		minStack: make([]int, 0),
	}
}

func (this *MinStack) Push(val int) {
	this.stack = append(this.stack, val)
	if len(this.minStack) == 0 || val < this.minStack[len(this.minStack)-1] {
		this.minStack = append(this.minStack, val)
	} else {
		this.minStack = append(this.minStack, this.minStack[len(this.minStack)-1])
	}
}

func (this *MinStack) Pop() {
	this.stack = this.stack[:len(this.stack)-1]
	this.minStack = this.minStack[:len(this.minStack)-1]
}

func (this *MinStack) Top() int {
	return this.stack[len(this.stack)-1]
}

func (this *MinStack) GetMin() int {
	return this.minStack[len(this.minStack)-1]
}

type TreeNode struct {
	Left, Right *TreeNode
	Val         int
}

/*
*
给你二叉树的根节点 root ，返回其节点值的 层序遍历 。
*/
func levelOrder(root *TreeNode) (ans [][]int) {
	queue := make([]*TreeNode, 0)
	if root != nil {
		queue = append(queue, root)
	}
	for len(queue) > 0 {
		size := len(queue)
		res := make([]int, 0, size)
		for i := 0; i < size; i++ {
			temp := queue[0]
			queue = queue[1:]
			res = append(res, temp.Val)
			if temp.Left != nil {
				queue = append(queue, temp.Left)
			}
			if temp.Right != nil {
				queue = append(queue, temp.Right)
			}
		}
		ans = append(ans, res)
	}
	return
}

func reverseBetween(head *ListNode, left int, right int) *ListNode {
	dummy := &ListNode{Next: head}
	cur := dummy
	var pre, p1, p2 *ListNode
	for i := 0; i < left; i++ {
		if i == left-1 {
			p1 = cur
		}
		cur = cur.Next
	}
	p2 = cur
	for i := 0; i <= right-left; i++ {
		nxt := cur.Next
		cur.Next = pre
		pre, cur = cur, nxt
	}
	p1.Next = pre
	p2.Next = cur
	return dummy.Next
}

/*
*
给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。

不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。

元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
*/
func removeElement(nums []int, val int) int {
	k := 0
	for i := 0; i < len(nums)-k; {
		if nums[i] == val {
			nums[len(nums)-k-1], nums[i] = nums[i], nums[len(nums)-k-1]
			k++
		} else {
			i++
		}
	}
	return len(nums) - k
}

/*
*
给你一个按 非递减顺序 排序的整数数组 nums，返回 每个数字的平方 组成的新数组，要求也按 非递减顺序 排序。
最大的一定在数组两边: 双指针
*/
func sortedSquares(nums []int) []int {
	newNums := make([]int, len(nums))
	insLoc := len(nums) - 1
	left, right := 0, len(nums)-1
	for left <= right {
		if nums[left]*nums[left] <= nums[right]*nums[right] {
			newNums[insLoc] = nums[right] * nums[right]
			right--
		} else {
			newNums[insLoc] = nums[left] * nums[left]
			left++
		}
		insLoc--
	}
	return newNums
}

/*
*
给定一个含有 n 个正整数的数组和一个正整数 target 。

找出该数组中满足其总和大于等于 target 的长度最小的 连续
子数组[numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。
*/
func minSubArrayLen(target int, nums []int) int {
	left := 0
	ans := math.MaxInt
	sum := 0
	for right := 0; right < len(nums); right++ {
		sum += nums[right]
		for sum >= target {
			ans = min(ans, right-left+1)
			sum -= nums[left]
			left++
		}
	}
	if ans == math.MaxInt {
		return 0
	}
	return ans
}

// 把图画出来就可以了
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

func reverseWords(s string) string {
	words := make([]string, 0)
	for i := 0; i < len(s); {
		if s[i] == ' ' {
			i++
		} else {
			j := i
			for ; j < len(s); j++ {
				if s[j] == ' ' {
					break
				}
			}
			words = append(words, s[i:j])
			i = j
		}
	}
	slices.Reverse(words)
	return strings.Join(words, " ")
}
func repeatedSubstringPattern(s string) bool {
	//枚举子串的长度
	for i := 1; i <= len(s)/2; i++ {
		if len(s)%i == 0 {
			flag := true
			//如果子串长度为3 需要每三个字符比较
			for j := 0; j+2*i <= len(s); j += i {
				if s[j:j+i] != s[j+i:j+2*i] {
					flag = false
					break
				}
			}
			if flag {
				return true
			}
		}
	}
	return false
}

func isValid(s string) bool {
	stack := make([]int32, 0, 1000)
	for _, char := range s {
		if strings.Contains("([{", string(char)) {
			stack = append(stack, char)
		} else if len(stack) != 0 {
			top := stack[len(stack)-1]
			if (top == '(' && char == ')') || (top == '[' && char == ']') || top == '{' && char == '}' {
				stack = stack[:len(stack)-1]
			} else {
				return false
			}
		} else {
			return false
		}
	}
	return len(stack) == 0
}

func removeDuplicates(s string) string {
	stack := make([]byte, 0, 1000)
	for _, char := range s {
		if len(stack) == 0 || byte(char) != stack[len(stack)-1] {
			stack = append(stack, byte(char))
		} else {
			stack = stack[:len(stack)-1]
		}
	}
	return string(stack)
}

func evalRPN(tokens []string) int {
	stack := make([]int, 0, 1000)
	for _, token := range tokens {
		if strings.Contains("+-*/", token) {
			right := stack[len(stack)-1]
			left := stack[len(stack)-2]
			stack = stack[:len(stack)-2]
			temp := 0
			if token == "+" {
				temp = left + right
			} else if token == "-" {
				temp = left - right
			} else if token == "*" {
				temp = left * right
			} else {
				temp = left / right
			}
			stack = append(stack, temp)
		} else {
			num, _ := strconv.Atoi(token)
			stack = append(stack, num)
		}
	}
	return stack[len(stack)-1]
}

// 滑动窗口最大值
// 维护一个单调减的单调队列并时刻弹出已经出了当前滑动窗口的元素
func maxSlidingWindow(nums []int, k int) (ans []int) {
	queue := make([]int, 0)
	for index, num := range nums {
		//弹出出了当前窗口的
		if len(queue) > 0 && index-queue[0] >= k {
			queue = queue[1:]
		}
		//维护单调递减的队列
		//
		for len(queue) > 0 && nums[queue[len(queue)-1]] < num {
			queue = queue[:len(queue)-1]
		}
		queue = append(queue, index)
		if index >= k-1 {
			ans = append(ans, nums[queue[0]])
		}
	}
	return
}

type Pair struct {
	value int
	count int
}

type PairHeap []Pair

func (p *PairHeap) Len() int {
	return len(*p)
}

func (p *PairHeap) Less(i, j int) bool {
	return (*p)[i].count > (*p)[j].count
}

func (p *PairHeap) Swap(i, j int) {
	(*p)[i], (*p)[j] = (*p)[j], (*p)[i]
}

func (p *PairHeap) Push(x any) {
	*p = append(*p, x.(Pair))
}

func (p *PairHeap) Pop() any {
	x := (*p)[len(*p)-1]
	*p = (*p)[:len(*p)-1]
	return x
}

// 前K个高频元素
func topKFrequent(nums []int, k int) (ans []int) {
	hp := &PairHeap{}
	count := make(map[int]int)
	for _, num := range nums {
		count[num]++
	}
	for value, cnt := range count {
		heap.Push(hp, Pair{value: value, count: cnt})
	}
	for i := 0; i < k; i++ {
		pair := heap.Pop(hp).(Pair)
		ans = append(ans, pair.value)
	}
	return
}

// 翻转二叉树
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

// 对称二叉树
func isSymmetric(root *TreeNode) bool {
	var dfs func(left, right *TreeNode) bool
	dfs = func(left, right *TreeNode) bool {
		//if left == nil && right == nil {
		//	return true
		//}
		//if left == nil || right == nil {
		//	return false
		//}
		if left == nil || right == nil {
			return left == right
		}
		return left.Val == right.Val && dfs(left.Left, right.Right) && dfs(left.Right, right.Left)
	}
	return dfs(root.Left, root.Right)
}

// 二叉树的最大深度
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

// 二叉树的最小深度
func minDepth(root *TreeNode) int {
	var dfs func(node *TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		//如果是叶子结点
		//if node.Left == nil && node.Right == nil {
		//	return 1
		//}
		//如果左子树为空 应该递归右子树
		//if node.Left == nil {
		//	return dfs(node.Right) + 1
		//}
		//如果右子树为空 应该递归左子树
		//if node.Right == nil {
		//	return dfs(node.Left) + 1
		//}

		//一步到位其实不好理解 完整的在上面的注释
		if node.Left == nil || node.Right == nil {
			return dfs(node.Left) + dfs(node.Right) + 1
		}
		return min(dfs(node.Left), dfs(node.Right)) + 1
	}
	return dfs(root)
}

// 用层序遍历实现二叉树的最小深度
func minDepth2(root *TreeNode) int {
	queue := make([]*TreeNode, 0, 1000)
	if root != nil {
		queue = append(queue, root)
	}
	height := 0
	for len(queue) > 0 {
		height++
		size := len(queue)
		for i := 0; i < size; i++ {
			temp := queue[0]
			queue = queue[1:]
			if temp.Left == nil && temp.Right == nil {
				return height
			}
			if temp.Left != nil {
				queue = append(queue, temp.Left)
			}
			if temp.Right != nil {
				queue = append(queue, temp.Right)
			}
		}
	}
	return height
}

// 完全二叉树的节点个数
// 先求二叉树的深度 然后看最后一层有多少节点
// 可以用层序遍历求解 时间复杂度和空间复杂度都比较高
//
//	func countNodes(root *TreeNode) int {
//		if root == nil {
//			return 0
//		}
//		queue := make([]*TreeNode, 0, 1000)
//		levels := make([][]int, 0)
//		queue = append(queue, root)
//		for len(queue) > 0 {
//			size := len(queue)
//			res := make([]int, 0, size)
//			for i := 0; i < size; i++ {
//				temp := queue[0]
//				queue = queue[1:]
//				res = append(res, temp.Val)
//				if temp.Left != nil {
//					queue = append(queue, temp.Left)
//				}
//				if temp.Right != nil {
//					queue = append(queue, temp.Right)
//				}
//			}
//			levels = append(levels, res)
//		}
//		return int(math.Pow(2, float64(len(levels)-1))) + len(levels[len(levels)-1]) - 1
//	}

func countNodes(root *TreeNode) int {
	var dfs func(node *TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		var leftHeight, rightHeight int
		temp := node.Left
		for ; temp != nil; temp = temp.Left {
			leftHeight++
		}
		temp = node.Right
		for ; temp != nil; temp = temp.Right {
			rightHeight++
		}
		if leftHeight == rightHeight {
			return int(math.Pow(2, float64(leftHeight+1))) - 1
		}
		return countNodes(node.Left) + countNodes(node.Right) + 1
	}
	return dfs(root)
}
func Abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// 平衡二叉树 两次递归 比较麻烦
//func isBalanced(root *TreeNode) bool {
//	var dfs func(node *TreeNode) bool
//	dfs = func(node *TreeNode) bool {
//		if node == nil {
//			return true
//		}
//		if Abs(maxDepth(node.Left)-maxDepth(node.Right)) <= 1 {
//			return dfs(node.Left) && dfs(node.Right)
//		} else {
//			return false
//		}
//	}
//	return dfs(root)
//}

func isBalanced(root *TreeNode) bool {
	var dfs func(node *TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		leftHeight := dfs(node.Left)
		if leftHeight == -1 {
			return -1
		}
		rightHeight := dfs(node.Right)
		if rightHeight == -1 || Abs(leftHeight-rightHeight) > 1 {
			return -1
		}
		return max(leftHeight, rightHeight) + 1
	}
	return dfs(root) != -1
}

// 二叉树的所有路径 初阶回溯
//func binaryTreePaths(root *TreeNode) (ans []string) {
//	var dfs func(node *TreeNode, path []string)
//	dfs = func(node *TreeNode, path []string) {
//		if node == nil {
//			return
//		}
//		path = append(path, strconv.Itoa(node.Val))
//		if node.Left == nil && node.Right == nil {
//			ans = append(ans, strings.Join(path, "->"))
//		}
//		dfs(node.Left, path)
//		dfs(node.Right, path)
//	}
//	dfs(root, []string{})
//	return
//}

// 左叶子之和
//func sumOfLeftLeaves(root *TreeNode) int {
//	var dfs func(node *TreeNode) int
//	dfs = func(node *TreeNode) int {
//		if node == nil {
//			return 0
//		}
//		//若左子树是叶子结点
//		if node.Left != nil && node.Left.Left == nil && node.Left.Right == nil {
//			return node.Left.Val + dfs(node.Right)
//		}
//		return dfs(node.Left) + dfs(node.Right)
//	}
//	return dfs(root)
//}

func sumOfLeftLeaves(root *TreeNode) (ans int) {
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		//左子树是叶子结点
		if node.Left != nil && node.Left.Left == nil && node.Left.Right == nil {
			ans += node.Left.Val
		}
		dfs(node.Left)
		dfs(node.Right)
	}
	dfs(root)
	return
}

// 正整数和负整数的最大计数
// 找到第一个>=0的 找到第一个>0的
func maximumCount(nums []int) int {
	negative := sort.SearchInts(nums, 0)
	positive := sort.Search(len(nums)-negative, func(i int) bool {
		return nums[i+negative] > 0
	})
	return max(len(nums)-negative-positive, negative)
}

// 找树左下角的值 层序遍历解法
//func findBottomLeftValue(root *TreeNode) int {
//	levels := levelOrder(root)
//	return levels[len(levels)-1][0]
//}

//提前得到树的最大深度 第一个达到最大深度的一定是左下角
//func findBottomLeftValue(root *TreeNode) int {
//	depth := maxDepth(root)
//	ans := math.MaxInt
//	var dfs func(node *TreeNode, height int)
//	dfs = func(node *TreeNode, height int) {
//		if node == nil || ans != math.MaxInt {
//			return
//		}
//		if height < depth {
//			dfs(node.Left, height+1)
//			dfs(node.Right, height+1)
//		} else if height == depth {
//			ans = node.Val
//			return
//		} else {
//			return
//		}
//	}
//	dfs(root, 1)
//	return ans
//}

// 如果深度变大 第一个到达的一定是最左边的节点
// 深度不变大 维持不变
func findBottomLeftValue(root *TreeNode) (ans int) {
	curMaxDepth := 0
	var dfs func(node *TreeNode, depth int)
	dfs = func(node *TreeNode, depth int) {
		if node == nil {
			return
		}
		if depth > curMaxDepth {
			ans = node.Val
			curMaxDepth = depth
		}
		dfs(node.Left, depth+1)
		dfs(node.Right, depth+1)
	}
	dfs(root, 1)
	return ans
}

// 路径总和
func hasPathSum(root *TreeNode, targetSum int) bool {
	var dfs func(node *TreeNode, sum int) bool
	dfs = func(node *TreeNode, sum int) bool {
		if node == nil {
			return false
		}
		sum += node.Val
		if node.Left == nil && node.Right == nil {
			return sum == targetSum
		}
		return dfs(node.Left, sum) || dfs(node.Right, sum)
	}
	return dfs(root, 0)
}

// 从中序和后序遍历序列构造二叉树
func buildTree(inorder []int, postorder []int) *TreeNode {
	if len(postorder) == 0 {
		return nil
	}
	//后序遍历的最后一个作为 "根节点"
	rootVal := postorder[len(postorder)-1]
	index := slices.Index(inorder, rootVal)
	return &TreeNode{
		Val:   rootVal,
		Left:  buildTree(inorder[:index], postorder[:index]),
		Right: buildTree(inorder[index+1:], postorder[index:len(postorder)-1]),
	}
}
